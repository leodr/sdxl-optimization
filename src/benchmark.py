from models import SDXLBase

model = SDXLBase()
model.prepare()
model.warm_up()

_, timings = model.generate_timed("A photograph of an astronaut riding a horse on the moon, highly detailed")
print(timings.format())
