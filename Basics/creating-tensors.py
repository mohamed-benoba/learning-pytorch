import torch
import numpy as np

data = [[1,2,5],[3,4,5]]
tensordata = torch.tensor(data)
print(tensordata.shape)

numpydata = np.array(data)
tensordatanp = torch.from_numpy(numpydata)
# print(tensordatanp)


data_ones = torch.ones_like(data, dtype=torch.float)
print(data_ones)