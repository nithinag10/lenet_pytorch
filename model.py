import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import time

device = 'cpu'

# Reading data

x_train = pd.read_csv('train.csv').to_numpy()
x_train_label = x_train[:,0]
x_train = x_train[:,1:]
x_test = pd.read_csv('DIG-MNIST.csv').to_numpy()
x_test_label = x_test[:,0]
x_test = x_test[:,1:]

# reshaping the data
x_train = x_train.reshape(-1, 1, 28, 28)
x_test = x_test.reshape(-1,1,28,28)

def normalize_MNIST_images(x):
    '''
    Args:
        x: data
    '''
    x_norm = x.astype(np.float32)
    return x_norm*2/255-1
# normalizing


x_train = normalize_MNIST_images(x_train)
x_test = normalize_MNIST_images(x_test)

# getting everything to tensor

ltest = torch.from_numpy(x_test_label)
ltrain = torch.from_numpy(x_train_label)
xtrain = torch.from_numpy(x_train)
xtest = torch.from_numpy(x_test)
#
class Model_LeNet(nn.Module):
    def __init__(self):
        super(Model_LeNet , self).__init__()
        self.conv1 = nn.Conv2d(1 ,6 , 5 , padding=2)
        self.conv2 = nn.Conv2d(6 , 16 , 5 )
        self.fc1 = nn.Linear(5*5*16 , 120)
        self.fc2 = nn.Linear(120 , 84)
        self.fc3 = nn.Linear(84 , 10)

    def forward(self , x):
        x = F.max_pool2d(F.relu(self.conv1(x)) , (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)) , (2,2))
        x = x.view(-1 , self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):

        size = x.size()[1:]
        return np.prod(size)


model = Model_LeNet()

# #
# # with torch.no_grad():
# #     yinit = model(xtest)
# #
# # _, lpred = yinit.max(1)
# # print(100 * (ltest == lpred).float().mean())
#
# def backprop(Batch , learning_rate, x_train, x_label , model , epochs):
#     N= x_train.size()[0]  # Training set size
#     NB = N // Batch
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.SGD(model.parameters() , lr=learning_rate)
#
#     for epoch in range(epochs):
#         running_loss = 0
#         shuffled_indices = np.random.permutation(NB)
#         for k in range(NB):
#             minibatch_indices = range(shuffled_indices[k] * Batch, (shuffled_indices[k] + 1) * Batch)
#             inputs = x_train[minibatch_indices]
#             labels = x_label[minibatch_indices]
#
#             # Initialize the gradients to zero
#             optimizer.zero_grad()
#
#             # Forward propagation
#             outputs = model(inputs)
#
#             # Error evaluation
#             loss = criterion(outputs, labels)
#
#             # Back propagation
#             loss.backward()
#
#             # Parameter update
#             optimizer.step()
#
#             # Print averaged loss per minibatch every 100 mini-batches
#             # Compute and print statistics
#             with torch.no_grad():
#                 running_loss += loss.item()
#             if k % 100 == 99:
#                 print('[%d, %5d] loss: %.3f' %
#                       (epoch + 1, k + 1, running_loss / 100))
#                 running_loss = 0.0
#
#
# start = time.time()
# backprop(Batch=75 , learning_rate=0.001 , x_train=xtrain , x_label=ltrain, model= model , epochs=30)
# end = time.time()
# print(f'It takes {end-start:.6f} seconds.')
#
# torch.save(model.state_dict(), 'kannada_mnist_new.pth')

model.load_state_dict(torch.load('kannada_mnist.pth'))
model.eval()

y = model(xtest)
print(100 * ((ltest==y.max(1)[1]).float().mean()))







