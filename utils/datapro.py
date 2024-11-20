import torch

def data2tensor(data):
    data1 = torch.stack(data, dim=0)
    data2 = data1.transpose(0, 1)
    data3 = data2.reshape(data2.shape[0] * data2.shape[1], *data2.shape[2:])
    return data3
