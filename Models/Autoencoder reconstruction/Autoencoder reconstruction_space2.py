
"""
"""
import os
import torch
import matplotlib.pyplot as pltlt
import random
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data.dataset import Subset
import torch.utils.data as data
import matplotlib.pyplot as plt

from tqdm import tqdm
import numpy as np
import scipy
# Define paths

data_root_dir = './datasets'

#%%
# I created a very simple 3-fold method to use during the training
def kFolded(dataset, batchsize = 512):
    
    random_list_indx = (list(range(0, len(dataset))))
    #random.shuffle(random_list_indx)
    
    dataloader_val_1 = DataLoader(Subset(dataset, random_list_indx[0:int(len(dataset)/3)]), 
                                             batch_size=batchsize, shuffle=True);
    
    dataloader_val_2 = DataLoader(Subset(dataset, random_list_indx[int(len(dataset)/3): int(len(dataset)/3)*2]), 
                             batch_size=batchsize, shuffle=True);
    
    dataloader_val_3 = DataLoader(Subset(dataset, random_list_indx[int((len(dataset)/3))*2:len(dataset)]), 
                            batch_size=batchsize, shuffle=True);
                                  
    dataloader_train_1 = DataLoader(Subset(dataset, random_list_indx[int(len(dataset)/3):len(dataset)]), 
                            batch_size=batchsize, shuffle=True);
    list_tmp = []
    list_tmp.append(Subset(dataset, random_list_indx[0:int(len(dataset)/3)]))
    list_tmp.append(Subset(dataset, random_list_indx[int(len(dataset)/3):len(dataset)]))
                                                        
    dataloader_train_2 = DataLoader(data.ConcatDataset(list_tmp), batch_size=batchsize, shuffle=True);
    
    dataloader_train_3 = DataLoader(Subset(dataset, random_list_indx[0:int(len(dataset)/3)*2]), 
                            batch_size=batchsize, shuffle=True);
    toReturn = [
                [dataloader_train_1, dataloader_val_1],  
                [dataloader_train_2, dataloader_val_2], 
                [dataloader_train_3,dataloader_val_3]
                ]
    return toReturn



#%%

#guard = np.random.randint(low = 0, high = 5)

class Occlusion():
    # Occlusion on an image
    def __init__(self, img_size):
        self.target_dims = img_size
        
    def __call__(self, sample):
        sample = np.asarray(sample, dtype = np.float64)
        guard = 0
        if(guard == 0):
            max_border = 28
            min_border = 8
            high_border = 14
            a = 0
            b = 0
            c = 0
            d = 0
            while(True):
                a = np.random.randint(max_border)
                b = np.random.randint(max_border)
                
                c = np.random.randint(low = min_border, high = high_border)
                d = np.random.randint(low = min_border, high = high_border)
                ac = a+c
                bd = b+d
                if (ac < max_border &
                    bd < max_border):
                    break
                
            occluded_img = sample.copy()
            
            for x in range(c):
                for y in range(d):
                    occluded_img[a+x,b+y] = 0.0        
            sample = occluded_img
        
        return sample
    
#%%
class GaussNoise():
    # Adds noise to an image
    
    def __init__(self, img_size):
        self.target_dims = img_size
        
    def __call__(self, sample):
        #sample = np.asarray(sample, dtype = np.float64)
        guard = np.random.randint(low = 0, high = 5)
        if(guard == 0):
            # Generate a matrix of noise
            
            noise1 = np.random.normal(0, 0.05, size=self.target_dims)*255
            # Add noise to the image
            #noise1 = np.asarray(noise1)
            noisy_target = sample + noise1


            # Return the noisy image
            sample = noisy_target
        
        return sample
     
        
class ToTensor():
    
    def __call__(self, sample):
        # Convert  to pytorch tensor
        sample = np.asarray(sample, dtype = np.int32) / 255
        tensor_sample = transforms.functional.to_tensor(sample).float()
        return tensor_sample
#%% Create dataset

train_transform = transforms.Compose([
    transforms.ToTensor(),
    ])

test_transform = transforms.Compose([
    Occlusion((28,28)),
    GaussNoise((28,28)),
    ToTensor(),
    ])

train_dataset = MNIST(data_root_dir, train=True,  download=True, transform = train_transform)
test_dataset  = MNIST(data_root_dir, train=False, download=True, transform = test_transform)

### Plot some sample
plt.close('all')
fig, axs = plt.subplots(5, 5, figsize=(8,8))
for ax in axs.flatten():
    img, label = random.choice(train_dataset)
    ax.imshow(img.squeeze().numpy(), cmap='gist_gray')
    ax.set_title('Label: %d' % label)
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()


#%%
# Define the network architecture
    
class Autoencoder(nn.Module):
    
    def __init__(self, encoded_space_dim):
        super().__init__()
        
        ### Encoder
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True)
        )
        self.encoder_lin = nn.Sequential(
            nn.Linear(3 * 3 * 32, 64),
            nn.ReLU(True),
            nn.Linear(64, encoded_space_dim)
        )
        
        ### Decoder
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 3 * 3 * 32),
            nn.ReLU(True)
        )
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1)
        )

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def encode(self, x):
        # Apply convolutions
        x = self.encoder_cnn(x)
        # Flatten
        x = x.view([x.size(0), -1])
        # Apply linear layers
        x = self.encoder_lin(x)
        return x
    
    def decode(self, x):
        # Apply linear layers
        x = self.decoder_lin(x)
        # Reshape
        x = x.view([-1, 32, 3, 3])
        # Apply transposed convolutions
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x

### Initialize the network
encoded_space_dim = 2
net = Autoencoder(encoded_space_dim=encoded_space_dim)
plt.savefig('sample_of_train_dataset')

### Some examples
# Take an input image (remember to add the batch dimension)
img = test_dataset[0][0].unsqueeze(0)
print('Original image shape:', img.shape)
# Encode the image
img_enc = net.encode(img)
print('Encoded image shape:', img_enc.shape)
# Decode the image
dec_img = net.decode(img_enc)
print('Decoded image shape:', dec_img.shape)


#%% Prepare training

### Define dataloader
train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=512, shuffle=False)

### Define a loss function
loss_fn = torch.nn.MSELoss()

### Define an optimizer
# lr = 1e-3 # Learning rate
# optim = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)

### If cuda is available set the device to GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
# Move all the network parameters to the selected device (if they are already on that device nothing happens)



#%% Network training

### Training function
def train_epoch(net, dataloader, loss_fn, optimizer, print_loss = True):
    # Training
    net.train()
    for sample_batch in dataloader:
        # Extract data and move tensors to the selected device
        image_batch = sample_batch[0].to(device)
        # Forward pass
        output = net(image_batch)
        loss = loss_fn(output, image_batch)
        # Backward pass
        optim.zero_grad()
        loss.backward()
        optim.step()
        # Print loss
        if print_loss:
            print('\t partial train loss: %f' % (loss.data))


### Testing function
def test_epoch(net, dataloader, loss_fn, optimizer):
    # Validation
    net.eval() # Evaluation mode (e.g. disable dropout)
    with torch.no_grad(): # No need to track the gradients
        conc_out = torch.Tensor().float()
        conc_label = torch.Tensor().float()
        for sample_batch in dataloader:
            # Extract data and move tensors to the selected device
            image_batch = sample_batch[0].to(device)
            # Forward pass
            out = net(image_batch)
            # Concatenate with previous outputs
            conc_out = torch.cat([conc_out, out.cpu()])
            conc_label = torch.cat([conc_label, image_batch.cpu()]) 
        # Evaluate global loss
        val_loss = loss_fn(conc_out, conc_label)
    return val_loss.data


# The best Net is #1
# [4, 0.01]
#%% Network analysis
encoded_space_dim = 2
lrate= 0.001

net = Autoencoder(encoded_space_dim=encoded_space_dim)
net.to(device)
optim = torch.optim.Adam(net.parameters(), lr=lrate,  weight_decay=1e-5)
training = False
num_epochs = 53
vall_loss_tmp = 100
counter = 1
if training:
       
    for epoch in range(num_epochs):
        print('EPOCH %d/%d' % (epoch + 1, num_epochs))
        ### Training
        train_epoch(net, dataloader=train_dataloader, loss_fn=loss_fn, optimizer=optim) 
        val_loss = test_epoch(net, dataloader=test_dataloader, loss_fn=loss_fn, optimizer=optim) 
        
        ### Plot progress
        img = test_dataset[0][0].unsqueeze(0).to(device)
        net.eval()
        with torch.no_grad():
            rec_img  = net(img)
        fig, axs = plt.subplots(1, 2, figsize=(12,6))
        axs[0].imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
        axs[0].set_title('Original image')
        axs[1].imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')
        axs[1].set_title('Reconstructed image (EPOCH %d)' % (epoch + 1))
        # plt.tight_layout()
        # plt.pause(0.1)
        # Save figure
        os.makedirs('autoencoder_progress_%d_features_final_2' % encoded_space_dim, exist_ok=True)
        plt.savefig('autoencoder_progress_%d_features_final_2/epoch_%d.png' % (encoded_space_dim, epoch + 1))
        plt.show()
        plt.close()
        torch.save(net.state_dict(), 'net_params_final2.pth')
        
    


#%%

#%%
    
net.load_state_dict(torch.load('net_params_final2.pth', map_location=device))
net.to(device)
val_loss = test_epoch(net, dataloader=test_dataloader, loss_fn=loss_fn, optimizer=optim) 
print(val_loss.data.item())
#%%
test_img_num = 1

img = test_dataset[test_img_num][0].unsqueeze(0).to(device)
net.eval()
with torch.no_grad():
    rec_img  = net(img)
fig, axs = plt.subplots(1, 2, figsize=(12,6))
axs[0].imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
axs[0].set_title('Original image #' + str(test_dataset[test_img_num][1]))
axs[1].imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')
axs[1].set_title('Reconstructed image (EPOCH %d)' % (50))