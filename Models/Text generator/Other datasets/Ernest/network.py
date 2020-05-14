# -*- coding: utf-8 -*-

from torch import nn


class Network(nn.Module):
    
    def __init__(self, input_size, hidden_units, layers_num, dropout_prob=0):
        # Call the parent init function (required!)
        super().__init__()
        # Define recurrent layer
        self.rnn = nn.LSTM(input_size=input_size, 
                           hidden_size=hidden_units,
                           num_layers=layers_num,
                           dropout=dropout_prob,
                           batch_first=True)
       
        # Define output layer
        self.linear = nn.Linear(hidden_units, input_size)
        
        
    def forward(self, x, state=None):
        x, rnn_state = self.rnn(x, state)
        x = self.linear(x)
        
        return x, rnn_state
    

def train_batch(net, batch_onehot, loss_fn, optimizer):
    
    ### Prepare network input and labels
    # Get the labels (the last letter of each sequence)
    labels_onehot = batch_onehot[:, -1, :]
    labels_numbers = labels_onehot.argmax(dim=1)
    # Remove the labels from the input tensor
    net_input = batch_onehot[:, :-1, :]
    # batch_onehot.shape =   [50, 100, 38]
    # labels_onehot.shape =  [50, 38]
    # labels_numbers.shape = [50]
    # net_input.shape =      [50, 99, 38]
    
    ### Forward pass
    # Eventually clear previous recorded gradients
    optimizer.zero_grad()
    # Forward pass
    net_out, _ = net(net_input)
    
    ### Update network
    # Evaluate loss only for last output
    loss = loss_fn(net_out[:, -1, :], labels_numbers)
    # Backward pass
    loss.backward()
    # Update
    optimizer.step()
    # Return average batch loss
    return float(loss.data)

def val_batch(net, batch_onehot, loss_fn, optimizer):
    
    ### Prepare network input and labels
    # Get the labels (the last letter of each sequence)
    labels_onehot = batch_onehot[:, -1, :]
    labels_numbers = labels_onehot.argmax(dim=1)
    # Remove the labels from the input tensor
    net_input = batch_onehot[:, :-1, :]
    # batch_onehot.shape =   [50, 100, 38]
    # labels_onehot.shape =  [50, 38]
    # labels_numbers.shape = [50]
    # net_input.shape =      [50, 99, 38]
    
    ### Forward pass
    # Eventually clear previous recorded gradients
    optimizer.zero_grad()
    # Forward pass
    net_out, _ = net(net_input)
    
    ### Update network
    # Evaluate loss only for last output
    loss = loss_fn(net_out[:, -1, :], labels_numbers)

    return float(loss.data)
#%%

# length of the alphabet 
alphabet_len = 34

if __name__ == '__main__':
    
    #%% Initialize network
    input_size = alphabet_len
    hidden_units = 128
    layers_num = 2
    dropout_prob = 0.3
    net = Network(input_size, hidden_units, layers_num, dropout_prob)
    
    #%% Get some real input from dataset
    
    from torch.utils.data import DataLoader
    from earnest_dataset import EarnestDataset, RandomCrop, OneHotEncoder, ToTensor
    from torchvision import transforms

    filepath = 'ernest.txt'
    crop_len = 20
    alphabet_len = alphabet_len
    trans = transforms.Compose([RandomCrop(crop_len),
                                OneHotEncoder(alphabet_len),
                                ToTensor()
                                ])
    dataset = EarnestDataset(filepath,crop_len=120, transform=trans)
    
    dataloader = DataLoader(dataset, batch_size=52, shuffle=True)
    
    for batch_sample in dataloader:
        batch_onehot = batch_sample['encoded_onehot']
        print(batch_onehot.shape)
        
        
    #%% Test the network output

    out, rnn_state = net(batch_onehot)
    print(out.shape)
    print(rnn_state[0].shape)
    print(rnn_state[1].shape)
        
    #%% Test network update
    
    import torch    
    optimizer = torch.optim.RMSprop(net.parameters())
    loss_fn = nn.CrossEntropyLoss()
    
    train_batch(net, batch_onehot, loss_fn, optimizer)
        
