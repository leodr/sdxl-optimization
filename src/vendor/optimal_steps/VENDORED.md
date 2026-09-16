# Vendored: OptimalSteps (OSS)

Source: https://github.com/bebebe666/OptimalSteps, commit `ee350436c86e29088c0cf550300a2e6a5911a7ee`
(2025-04-13). Paper: Pei, Hu, Gu, "Optimal Stepsize for Diffusion Sampling", arXiv:2503.21774. MIT license,
see `LICENSE`.

Copied unmodified: `OSS/` (the search and inference code), `LICENSE`, `README.md`. Not copied: `examples/`
and `scripts/` (DiT, FLUX, Open-Sora and Wan2.1 integrations) and `teaser.png`.

Do not edit the files here. The SDXL adaptation lives in `src/oss_sdxl.py`, which uses `cal_medium` and
`infer_OSS` from `OSS/OSS.py` as they are, and reimplements the search (`search_OSS`) with the changes
listed in its docstring.
