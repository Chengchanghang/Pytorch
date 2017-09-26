import torch
import torch.autograd import Variable
import matplotlib.pyplot as plt
x = torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
y = x.pow(2)
x,y = Variable(x),Variable(y)

x = x.data.numpy()
y = y.data.numpy()

plt.plot(x,y)
plt.show()