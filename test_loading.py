import torch

model = torch.load('last.pt')['model']

f = open("demofile.txt", "a")
for name, param in model.named_parameters():
    f.write('\n' + str(name))
    f.write('\n' + str(param))
f.close()

