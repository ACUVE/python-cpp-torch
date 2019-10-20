import torch
import torchvision
import time

img = torch.ones((1, 3, 244, 244), device="cuda")
img.requires_grad_(False)

model = torchvision.models.resnet18(pretrained=True)
model.eval()
model.to("cuda")

NUM_OF_LOOP = 1000

# warmup
for _ in range(10):
    model(img)

sum_of_duration = 0.0
for _ in range(NUM_OF_LOOP):
    start = time.time()
    ret = model(img)
    end = time.time()

    duration = end - start
    sum_of_duration += duration

print(f"{sum_of_duration / NUM_OF_LOOP * 1000:.2f}ms/frame, {NUM_OF_LOOP / sum_of_duration:.2f}fps")

model2 = torch.jit.load("traced_resnet_model.pt")
model2.eval()
model2.to("cuda")

# warmup
for _ in range(10):
    model2(img)

sum_of_duration = 0.0
for _ in range(NUM_OF_LOOP):
    start = time.time()
    ret2 = model2(img)
    end = time.time()

    duration = end - start
    sum_of_duration += duration

print(f"{sum_of_duration / NUM_OF_LOOP * 1000:.2f}ms/frame, {NUM_OF_LOOP / sum_of_duration:.2f}fps")

print(torch.max((ret.cpu() - ret2.cpu()) / ret.cpu()))
