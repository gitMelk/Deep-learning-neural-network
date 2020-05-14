# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 23:26:48 2019

"""
#%% 
import torch
from torch import nn
import scipy.io
#%% 
# This function fix the vector of the labels from:
# [[1], [2], [3], [4], [5], [6]...] to
# [1,2,3,4,5,6...]
def fixarray(vector):
    myList = [];
    for i in range(len(vector)):
        myList.append(vector[i].item());
    return torch.LongTensor(myList)
# define the model
class Net(nn.Module):
       
    def __init__(self, Ni, Nh1, Nh2, No):
        super().__init__()
        self.fc1 = nn.Linear(in_features=Ni, out_features=Nh1)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = nn.Linear(Nh1, Nh2)
        self.relu2 = torch.nn.ReLU()
        self.fc3 = nn.Linear(Nh2, No)
        self.logSoftmax = nn.LogSoftmax(dim=1)    
           
    def forward(self, x, additional_out=False):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        output = self.logSoftmax(self.fc3(x))
        if additional_out:
            ps = torch.exp(output)
            probab = list(ps.cpu().detach().numpy()[0])
            pred_label = probab.index(max(probab))
            return output, pred_label
        return output

# load the dataset
mat = scipy.io.loadmat('MNIST.mat')
x_all = mat["input_images"];
y_all = mat["output_labels"];

test_data = []
for i in range(len(x_all)):
   test_data.append([x_all[i], y_all[i]]);
   
testloader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)
num_test_points = len(test_data);

### Reload the network state
# First initialize the network
net = Net(784, 64, 128, 10) 
# Load the state dict previously saved
net_state_dict = torch.load('net_parameters.torch')
# Update the network parameters
net.load_state_dict(net_state_dict)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)
#%% Compute the model accuracy
correct_count, all_count = 0, 0
for images,labels in testloader:
    for i in range(len(labels)):
        labels = fixarray(labels);
        img = images[i].view(1, 784)
        # Turn off gradients to speed up this part
        with torch.no_grad():
            tmp, logps = net(img.to(device),additional_out = True)
        pred_label = logps
        true_label = labels.numpy()[i]
        if(true_label == pred_label):
          correct_count += 1
        all_count += 1
mod_accu_new = (correct_count/all_count);
print("\nModel Accuracy =", (correct_count/all_count))
