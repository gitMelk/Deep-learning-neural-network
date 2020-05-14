import numpy as np
import torch
import matplotlib.pyplot as plt
from time import time
import scipy.io
from torch import optim
from torch import nn
import random
from sklearn.model_selection import KFold
#%% Load the data and 
np.random.seed(7);

mat = scipy.io.loadmat('MNIST.mat')
x_all = mat["input_images"];
y_all = mat["output_labels"];

#Test dataset
test_data = []
for i in range(10000):
   test_data.append([x_all[i], y_all[i]]);
   
testloader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)
num_test_points = len(test_data);

# This function fix the vector of the labels from:
# [[1], [2], [3], [4], [5], [6]...] to
# [1,2,3,4,5,6...]
def fixarray(vector):
    myList = [];
    for i in range(len(vector)):
        myList.append(vector[i].item());
    return torch.LongTensor(myList)


#%% Model description

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
        # when additional_out return the single number prediction
        if additional_out:
            ps = torch.exp(output)
            probab = list(ps.cpu().detach().numpy()[0])
            pred_label = probab.index(max(probab))
            return output, pred_label
        return output
        
#%% Net paramiters
# Loss
criterion = nn.NLLLoss()
# input size, fixed
Ni = 784
# hidden sizes, can be changed
Nh1_grid = ["32", "64", "128"];
lr_grid = ["0.01", "0.001","0.0001"];
Nh2_grid = ["32", "64", "128"];
# output size, fixed
No = 10
#net = Net(Ni, Nh1, Nh2, No)
train_loss_log = []
test_loss_log = []
lr = 1e-2
time0 = time()
epochs = 10
#%% Kfold Cross Validation
# Define the split - into 3 folds

x_train_copy = [];
for i in range(10000):
   x_train_copy.append(x_all[i+10000]);
y_train_copy = [];
for i in range(10000):
   y_train_copy.append(y_all[i+10000]);
   
y_all[i+10000]
result = zip(x_train_copy, y_train_copy)
resultList = list(result)
random.shuffle(resultList)
x_train_s, y_train_s =  zip(*resultList)

kf = KFold(n_splits=3, shuffle=False)
print(kf)
X = np.asarray(x_train_s)
y = np.asarray(y_train_s)
kf.get_n_splits(X)
# Print the splits
for train_index, test_index in kf.split(X):
    print('TRAIN:', train_index, 'TEST:', test_index)
    X_train_kf, X_test_kf = X[train_index], X[test_index]
    y_train_kf, y_test_kf = y[train_index], y[test_index] 

loss_opt = 1000000000
#%%
mod_accu = 0;
tell_me_params = [];

for lr_tmp in lr_grid:    
    for N1 in Nh1_grid:
        for N2 in Nh2_grid:
                    
            # loss kfold
            loss_temp = [] 
            #
            for train_index, test_index in kf.split(X):
                # Net parameters
                Nh1 = int(N1);
                Nh2 = int(N2);
                net = Net(Ni, Nh1, Nh2, No)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                net.to(device)
                optimizer = optim.SGD(net.parameters(), lr=float(lr_tmp), momentum=0.5);
                # dataset used
                X_train_kf, X_test_kf = X[train_index], X[test_index];
                y_train_kf, y_test_kf = y[train_index], y[test_index];
                # temporal variables
                train_loss_log = []
                test_loss_log = []
                avg_test_loss = 0;
                train_data = [];
                val_data = [];
                # create the dalaloader
                for i in range(len(X_train_kf)):
                    train_data.append([X_train_kf[i], y_train_kf[i]]);
                for i in range(len(X_test_kf)):
                    val_data.append([X_test_kf[i], y_test_kf[i]]);
                trainloader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)
                valloader = torch.utils.data.DataLoader(val_data, batch_size=64, shuffle=True)
                for e in range(epochs):
                    running_loss_train = 0;
                    running_loss_val = 0;
                    # training
                    for images, labels in trainloader:
                        labels = fixarray(labels);
                        # Flatten MNIST images into a 784 long vector
                        images = images.view(images.shape[0], -1)
                        
                        net.train()   
                        optimizer.zero_grad()
                        
                        output = net(images.to(device))
                        loss = criterion(output, labels.to(device))
                        loss.backward()
                        
                        optimizer.step()
                        running_loss_train += loss.item()
                    # validation
                    for images, labels in valloader:
                        labels = fixarray(labels);
                        output = net(images.to(device))
                        loss = criterion(output, labels.to(device))
                        running_loss_val += loss.item()
                        if ( e == epochs-1 ):
                            loss_temp.append(running_loss_val/len(valloader))
                    else:
                        #print("Epoch {} - Validation loss: {}".format(e, running_loss_val/len(valloader)))
                        print('Epoch %d - Train loss: %.5f - Test loss: %.5f' % (e + 1,  float(running_loss_train/len(trainloader)), float(running_loss_val/len(valloader))))
                       
            # see if the network with the new parameters is better than before
            loss_avg = np.mean(np.array(loss_temp))
            if (loss_avg<=loss_opt):
                loss_opt = loss_avg
                Nh1_opt = Nh1
                Nh2_opt = Nh2
                tell_me_params = [lr_tmp,Nh1_opt,Nh2_opt]
            
        print("Values: " + str(lr_tmp) +","+str(Nh1)+"," +str(Nh2)+","+ str(loss_avg))
                    
        #del net
            
        
print("\nTraining Time (in minutes) =",(time()-time0)/60)
print(tell_me_params)

#%% Final training
epochs = 50;
lr_tmp = 0.01;
Ni = 784
Nh1 = 64;
Nh2 = 128;
No = 10
net = Net(Ni, Nh1, Nh2, No)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)
optimizer = optim.SGD(net.parameters(), lr=lr_tmp, momentum=0.5);
train_data = []
   
for i in range(50000):
   train_data.append([x_all[10000+i], y_all[10000+i]])   

num_train_points = len(train_data);

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

for e in range(epochs):
    running_loss_train = 0;
    for images, labels in trainloader:
        labels = fixarray(labels);
        images = images.view(images.shape[0], -1)
        
        net.train()   
        optimizer.zero_grad()
        
        output = net(images.to(device))
        loss = criterion(output, labels.to(device))
        loss.backward()
        
        optimizer.step()
        running_loss_train += loss.item()
    else:
        print("Epoch {} - Training loss: {}".format(e, running_loss_train/len(trainloader)))
#%%
net_state_dict = net.state_dict()
# Save the state dict to a file
torch.save(net_state_dict, 'net_parameters.torch')
#
#
##%%
#
#net = Net(784, 64, 128, 10) 
#net_state_dict = torch.load('net_parameters.torch')
#net.load_state_dict(net_state_dict)
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#net.to(device)


#%% Simple function to show results
def view_classify(img, ps):

    img = img.transpose(1, 2).flip(1)
    ps = ps.cpu().data.numpy().squeeze()
    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze(), cmap='Greys')
    ax1.invert_yaxis()
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    
    plt.xticks(rotation=0)
    plt.tight_layout()
    
#%% Print some results

images, labels = next(iter(testloader))
labels = fixarray(labels);
img = images[0].view(1, 784)
with torch.no_grad():
    logps = net(img.to(device))

ps = torch.exp(logps)
probab = list(ps.cpu().numpy()[0])
print("Predicted Digit =", probab.index(max(probab)))
view_classify(img.view(1, 28, 28), ps)
tes, test_result = net(img.to(device), additional_out=True)
print(test_result)

#%% Accuracy on the Test dataset
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
print("Number Of Images Tested =", all_count)
print("\nModel Accuracy =", (correct_count/all_count))
#%%
# Visualize the receptive field of hidden neurons
net_weights_1 = net.fc1.weight
weights_1 = net_weights_1.cpu().detach().numpy()
net_weights_2 = net.fc2.cpu().weight
weights_2_tmp = net_weights_2.cpu().detach().numpy()
net_weights_3 = net.fc3.cpu().weight
weights_3_tmp = net_weights_3.cpu().detach().numpy()

weights_2 = np.matmul(weights_2_tmp, weights_1)
weights_3 = np.matmul(weights_3_tmp, weights_2)
#%%
fig, ax = plt.subplots(2, 5, sharex='col', sharey='row')
counter = 0;
for i in range(2):
    for j in range(5):
        rf3_tmp = torch.from_numpy(weights_1[counter]).float().to(device).view(1, 784)
        rf3_tmp = rf3_tmp.to(device)
        rf3_tmp = rf3_tmp.view(1, 28, 28)
        rf3_tmp = rf3_tmp.transpose(1, 2).flip(1)
        ax[i, j].imshow(rf3_tmp.resize_(1, 28, 28).cpu().numpy().squeeze(),cmap='YlGnBu')
        counter = counter + 1

fig, ax = plt.subplots(2, 5, sharex='col', sharey='row')
counter = 0;
for i in range(2):
    for j in range(5):
        rf3_tmp = torch.from_numpy(weights_2[counter]).float().to(device).view(1, 784)
        rf3_tmp = rf3_tmp.to(device)
        rf3_tmp = rf3_tmp.view(1, 28, 28)
        rf3_tmp = rf3_tmp.transpose(1, 2).flip(1)
        ax[i, j].imshow(rf3_tmp.resize_(1, 28, 28).cpu().numpy().squeeze(),cmap='YlGnBu')
        counter = counter + 1

fig, ax = plt.subplots(2, 5, sharex='col', sharey='row')
counter = 0;
for i in range(2):
    for j in range(5):
        rf3_tmp = torch.from_numpy(weights_3[counter]).float().to(device).view(1, 784)
        rf3_tmp = rf3_tmp.to(device)
        rf3_tmp = rf3_tmp.view(1, 28, 28)
        rf3_tmp = rf3_tmp.transpose(1, 2).flip(1)
        ax[i, j].imshow(rf3_tmp.resize_(1, 28, 28).cpu().numpy().squeeze(),cmap='YlGnBu')
        counter = counter + 1


