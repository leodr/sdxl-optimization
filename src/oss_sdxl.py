"""Optimal Stepsize for Diffusion Sampling (OSS) for SDXL base 1.0, on the regular diffusers pipeline.

OSS (vendored in vendor/optimal_steps, see VENDORED.md) picks a student schedule of K steps out of a teacher
schedule of N steps: it runs the teacher once, then chooses the K teacher timesteps whose Euler steps stay
closest to the teacher trajectory, by dynamic programming. The result is a list of timesteps that the
unmodified pipeline takes via `pipe(..., timesteps=...)`.

How SDXL maps onto OSS. OSS needs a model returning a velocity v and a time grid `fm_steps` such that one
sampler step is `x_j = x_i + v * (fm_steps[j] - fm_steps[i])`. SDXL's default scheduler is Euler on an
epsilon-prediction model in sigma space, whose step is exactly `x_j = x_i + eps * (sigma_j - sigma_i)`.
So the wrapper returns the classifier-free-guided eps (as float32, rounded like the scheduler rounds it) and
uses the teacher's sigmas as `fm_steps`, and no
conversion to flow matching is needed. (Converting, as the vendored DiT wrapper does, gives the same steps:
Euler in sigma space and Euler in flow-matching time are both DDIM. It also gives the same search result,
since OSS only compares costs at the same time index, where the conversion scales all candidates equally.)

Because the student timesteps are a subset of the teacher's integer timesteps, EulerDiscreteScheduler looks
up the same sigmas for them, and `pipe(timesteps=student)` runs exactly the steps the search evaluated.
"""

import time

from diffusers.utils.torch_utils import randn_tensor
import torch

from vendor.optimal_steps.OSS.OSS import infer_OSS


def make_noise(pipe, seed: int, height: int = 1024, width: int = 1024) -> torch.Tensor:
    """The unscaled initial noise the pipeline draws for `generator=torch.Generator(device).manual_seed(seed)`."""
    shape = (1, pipe.unet.config.in_channels, height // pipe.vae_scale_factor, width // pipe.vae_scale_factor)
    generator = torch.Generator(pipe.device).manual_seed(seed)
    return randn_tensor(shape, generator=generator, device=pipe.device, dtype=pipe.unet.dtype)


class WrappedSDXL:
    """SDXL's U-Net with classifier-free guidance in the OSS model interface.

    Index i in 0..N addresses the teacher grid: i = N is the initial noise, i = 0 the final sample, and
    index i >= 1 is the state before teacher step N - i, denoised at timesteps[N - i] with sigmas[N - i].
    `fm_steps[i]` is that state's sigma (0 for i = 0).
    """

    def __init__(self, pipe, timesteps: torch.Tensor, sigmas: torch.Tensor, conditioning: dict[str, torch.Tensor],
                 guidance_scale: float):
        """timesteps (N) and sigmas (N + 1, ending in 0) are the scheduler's after set_timesteps for the teacher.
        conditioning holds prompt_embeds, add_text_embeds and add_time_ids as the pipeline passes them to the
        U-Net: negative and positive prompt concatenated along the batch dimension."""
        self.unet = pipe.unet
        self.n = len(timesteps)
        self.timesteps = timesteps
        self.fm_steps = torch.flip(sigmas, dims=[0])
        self.conditioning = conditioning
        self.guidance_scale = guidance_scale

    def __call__(self, x, t, y, kwargs):
        # Mirrors one iteration of StableDiffusionXLPipeline's denoising loop, up to the scheduler step.
        # y (class embedding) and kwargs are part of the OSS interface; the prompt is fixed in conditioning.
        assert x.shape[0] == 1, "one latent at a time; the conditioning holds a single prompt"
        i = int(t[0])
        sigma = self.fm_steps[i]
        latent_model_input = torch.cat([x] * 2) / ((sigma**2 + 1) ** 0.5)
        noise_pred = self.unet(
            latent_model_input,
            self.timesteps[self.n - i],
            encoder_hidden_states=self.conditioning["prompt_embeds"],
            added_cond_kwargs={"text_embeds": self.conditioning["add_text_embeds"],
                               "time_ids": self.conditioning["add_time_ids"]},
            return_dict=False,
        )[0]
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

        # The velocity is eps, but computed the way EulerDiscreteScheduler.step computes its derivative, so that
        # OSS's `x + v * dt` reproduces the scheduler bit for bit. Returning noise_pred directly differs by one
        # float16 rounding step per step, since the scheduler multiplies sigma * eps in float16.
        sample = x.float()
        pred_original_sample = sample - sigma * noise_pred
        return (sample - pred_original_sample) / sigma

    def to_pipeline_timesteps(self, oss_steps: list[int]) -> list[int]:
        """Teacher indices in OSS order (ascending, ending at N) to pipeline timesteps (descending)."""
        return [int(self.timesteps[self.n - i]) for i in reversed(oss_steps)]


@torch.no_grad()
def run_teacher(pipe, prompt: str, noise: torch.Tensor, teacher_steps: int = 50,
                guidance_scale: float = 5.0) -> tuple[WrappedSDXL, torch.Tensor]:
    """Run the pipeline with the teacher schedule and record every latent along the way.

    Returns the wrapped model and the trajectory as a tensor (N + 1, C, H, W), indexed like WrappedSDXL:
    trajectory[N] is the scaled initial noise, trajectory[0] the final latent.
    """
    latents, conditioning = [], {}

    def record(pipe, step, timestep, tensors):
        if step == 0:
            conditioning.update({k: tensors[k] for k in ("prompt_embeds", "add_text_embeds", "add_time_ids")})
        latents.append(tensors["latents"].clone())
        return {}

    # The pipeline multiplies the given latents by init_noise_sigma, as it does with latents it draws itself.
    pipe(prompt=prompt, num_inference_steps=teacher_steps, guidance_scale=guidance_scale, latents=noise.clone(),
         output_type="latent", callback_on_step_end=record,
         callback_on_step_end_tensor_inputs=["latents", "prompt_embeds", "add_text_embeds", "add_time_ids"])

    scheduler = pipe.scheduler
    model = WrappedSDXL(pipe, scheduler.timesteps.clone(), scheduler.sigmas.clone(), conditioning, guidance_scale)
    trajectory = torch.stack([(noise[0] * scheduler.init_noise_sigma).to(noise.dtype)] + [z[0] for z in latents])
    return model, torch.flip(trajectory, dims=[0])


@torch.no_grad()
def search_oss(model: WrappedSDXL, trajectory: torch.Tensor, student_steps: int) -> tuple[list[int], float]:
    """OSS dynamic programming: the K = student_steps teacher indices whose Euler steps best follow the trajectory.

    Follows vendor/optimal_steps/OSS/OSS.py:search_OSS (same cost, the mean squared distance to the teacher
    latent at the landing index, and the same rule of keeping, per step count and index, only the closest
    state), with these changes:

    - Only states actually reached are expanded. The original fills every index with the initial noise, so
      step 1 could start from pure noise at a lower noise level, and later steps from indices never reached.
    - Exactly K steps, ending at index 0. The original carries the previous step count's table forward, so
      the traceback can mix step counts.
    - Indices from which the remaining steps cannot reach 0 are skipped, and the last step only evaluates
      landing at 0. This saves U-Net calls and doesn't change the result.
    - All landing indices of one U-Net call are scored in one batched operation, as in search_OSS_batch.
    - The Euler update runs in float32 and is cast back to the latent dtype, like EulerDiscreteScheduler.step.

    Returns the indices in the vendored format (ascending, ending at N, e.g. [3, 8, ..., 50]) and the final
    cost, the mean squared distance between the student's and the teacher's final latent.
    """
    n, k_steps = model.n, student_steps
    assert 1 <= k_steps <= n
    device, dtype = trajectory.device, trajectory.dtype
    fm_steps = model.fm_steps.to(device=device, dtype=torch.float32)
    target = trajectory.float()

    # cost[k, j]: distance of the best state at index j reached in exactly k steps; parent[k, j]: where it came from.
    cost = torch.full((k_steps + 1, n + 1), float("inf"), device=device)
    parent = torch.full((k_steps + 1, n + 1), -1, dtype=torch.long, device=device)
    states = trajectory.clone()  # states[j]: that best state; only valid where cost[k, j] is finite
    cost[0, n] = 0.0

    for k in range(k_steps):
        next_states = states.clone()
        reached = torch.nonzero(torch.isfinite(cost[k])).flatten().tolist()
        for i in sorted(reached, reverse=True):
            # After this step, k_steps - k - 1 steps remain, each of which lowers the index by at least 1.
            lowest = k_steps - k - 1
            if i <= lowest:
                continue
            js = torch.arange(lowest, i, device=device) if lowest > 0 else torch.zeros(1, dtype=torch.long, device=device)
            z_i = states[i].unsqueeze(0)
            v = model(z_i, torch.tensor([i], device=device), None, {})
            dt = (fm_steps[js] - fm_steps[i]).view(-1, 1, 1, 1)
            z_js = (z_i.float() + v * dt).to(dtype)
            c = ((z_js.float() - target[js]) ** 2).mean(dim=(1, 2, 3))
            better = c < cost[k + 1, js]
            cost[k + 1, js[better]] = c[better]
            parent[k + 1, js[better]] = i
            next_states[js[better]] = z_js[better]
        states = next_states

    path, j = [], 0
    for k in range(k_steps, 0, -1):
        j = int(parent[k, j])
        path.append(j)
    assert path[-1] == n and path == sorted(path), path
    return path, float(cost[k_steps, 0])


@torch.no_grad()
def search_prompt(pipe, prompt: str, seed: int, student_steps: int = 10, teacher_steps: int = 50,
                  guidance_scale: float = 5.0) -> dict:
    """Teacher run plus OSS search for one prompt and seed. Returns the indices, timesteps, cost and timing."""
    start = time.perf_counter()
    model, trajectory = run_teacher(pipe, prompt, make_noise(pipe, seed), teacher_steps, guidance_scale)
    oss_steps, final_cost = search_oss(model, trajectory, student_steps)
    return {
        "prompt": prompt,
        "seed": seed,
        "oss_steps": oss_steps,
        "timesteps": model.to_pipeline_timesteps(oss_steps),
        "final_latent_mse": final_cost,
        "seconds": time.perf_counter() - start,
    }


@torch.no_grad()
def check_adaptation(pipe, prompt: str, seed: int, student_steps: int = 10, teacher_steps: int = 50,
                     guidance_scale: float = 5.0) -> dict[str, float]:
    """Check the wrapper against the pipeline, as max absolute latent differences (0 means identical):

    - teacher: the vendored infer_OSS with every teacher index, vs the pipeline's own teacher run
    - student: infer_OSS with the searched indices, vs pipe(timesteps=...) with the converted timesteps
    - student_cost: the search's reported final cost vs the mean squared distance of that pipeline run
    """
    noise = make_noise(pipe, seed)
    model, trajectory = run_teacher(pipe, prompt, noise, teacher_steps, guidance_scale)
    z_n = trajectory[-1:].clone()
    teacher = infer_OSS(list(range(1, teacher_steps + 1)), model, z_n, None, pipe.device)

    oss_steps, final_cost = search_oss(model, trajectory, student_steps)
    student = infer_OSS(oss_steps, model, z_n, None, pipe.device)
    pipeline_student = pipe(prompt=prompt, timesteps=model.to_pipeline_timesteps(oss_steps),
                            guidance_scale=guidance_scale, latents=noise.clone(), output_type="latent").images
    return {
        "teacher": float((teacher.float() - trajectory[0].float()).abs().max()),
        "student": float((student.float() - pipeline_student.float()).abs().max()),
        "student_cost": abs(final_cost - float(((pipeline_student[0].float() - trajectory[0].float()) ** 2).mean())),
    }
