import torch
import numpy as np

tensor = torch.ones(4,4)
tensor[:,-1] = 2
tensor[:,1] = 3
print(tensor)

