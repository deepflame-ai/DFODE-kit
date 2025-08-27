import torch

class MLP(torch.nn.Module):
    def __init__(self, layer_info):
        super(MLP, self).__init__()
        
        self.net = torch.nn.Sequential()
        n = len(layer_info) - 1
        for i in range(n - 1):
            self.net.add_module('linear_layer_%d' %(i), torch.nn.Linear(layer_info[i], layer_info[i + 1]))
            self.net.add_module('gelu_layer_%d' %(i), torch.nn.GELU())
        self.net.add_module('linear_layer_%d' %(n - 1), torch.nn.Linear(layer_info[n - 1], layer_info[n]))

    def forward(self, x):
        return self.net(x)