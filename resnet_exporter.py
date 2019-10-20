import torch
import torchvision

model = torchvision.models.resnet18(pretrained=True)
model.eval()
example = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("traced_resnet_model.pt")

img = torch.ones((1, 3, 244, 244))
model.eval()
ret1 = model(img)
traced_script_module.eval()
ret2 = traced_script_module(img)

print(torch.max((ret1 - ret2) / ret1))
