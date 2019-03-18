import torch
a = torch.ones(32, 10)
b = torch.ones(32, 11)
d = torch.ones(32, 40)
c = torch.cat([a, b, d], 1)
print(c.size())