"""
View more, visit my tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou

Dependencies:
torch: 0.1.11
matplotlib
"""
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import gzip 
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision      # 数据库模块
import matplotlib.pyplot as plt

torch.manual_seed(1)    # reproducible

def vectorized_result(y):
    z = np.zeros((10,))
    z[y] = 1.0 
    return z

f = gzip.open('mnist.pkl.gz')
tr_d, va_d, te_d = pickle.load(f,encoding='bytes')
f.close()

training_inputs = [np.reshape(x, (784, )) for x in tr_d[0]]
training_results = [vectorized_result(y) for y in tr_d[1]]
traing_data = zip(training_inputs, training_results)
validation_inputs = [np.reshape(x, (784, )) for x in va_d[0]]
validation_data = zip(validation_inputs, va_d[1])
test_inputs = [np.reshape(x, (784, )) for x in te_d[0]]
test_data = zip(test_inputs, te_d[1])
tr_data = []
for x,y in traing_data:
    x = torch.from_numpy(x)
    xy = (x,y)
    tr_data.append(xy)
    
class Net(torch.nn.Module):
    def __init__(self, input_size=784,hidden_size=500,num_classes=10):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(input_size, hidden_size)
    
        self.predict = torch.nn.Linear(hidden_size,num_classes)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x

net = Net()     # define the network
print(net) 
x,y = tr_data[1]
x = Variable(x)
b_y = torch.FloatTensor(y)
b_y = Variable(b_y)

optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
loss_func = torch.nn.MSELoss()
