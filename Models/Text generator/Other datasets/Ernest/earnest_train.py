# -*- coding: utf-8 -*-

import argparse
import torch
import json
from torch import optim, nn
from earnest_dataset import EarnestDataset, RandomCrop, OneHotEncoder, ToTensor
from network import Network, train_batch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
import random 
from torch.utils.data.dataset import Subset
import numpy as np
##############################
##############################
## PARAMETERS
##############################
parser = argparse.ArgumentParser(description='Train the chapter generator network.')


# Dataset
parser.add_argument('--datasetpath',    type=str,   default='ernest.txt',    help='Path of the train txt file')
parser.add_argument('--crop_len',       type=int,   default=40,    help='Number of input letters')
parser.add_argument('--alphabet_len',   type=int,   default=36,    help='Number of letters in the alphabet')

# Network
parser.add_argument('--hidden_units',   type=int,   default=256,    help='Number of RNN hidden units')
parser.add_argument('--layers_num',     type=int,   default=2,      help='Number of RNN stacked layers')
parser.add_argument('--dropout_prob',   type=float, default=0.3,    help='Dropout probability')

# Training
parser.add_argument('--batchsize',      type=int,   default=5000,   help='Training batch size')
parser.add_argument('--num_epochs',     type=int,   default=80000,    help='Number of training epochs')

# Save
parser.add_argument('--out_dir',     type=str,   default='ernest',    help='Where to save models and params')

##############################
##############################

def kFolded(dataset):
    
    random_list_indx = (list(range(0, len(dataset))))
#    random.shuffle(random_list_indx)
    
    dataloader_val_1 = DataLoader(Subset(dataset, random_list_indx[0:int(len(dataset)/3)]), 
                                             batch_size=args.batchsize, shuffle=True);
    
    dataloader_val_2 = DataLoader(Subset(dataset, random_list_indx[int(len(dataset)/3): int(len(dataset)/3)*2]), 
                             batch_size=args.batchsize, shuffle=True);
    
    dataloader_val_3 = DataLoader(Subset(dataset, random_list_indx[int((len(dataset)/3))*2:len(dataset)-1]), 
                            batch_size=args.batchsize, shuffle=True);
                                  
    dataloader_train_1 = DataLoader(Subset(dataset, random_list_indx[int(len(dataset)/3):len(dataset)-1]), 
                            batch_size=args.batchsize, shuffle=True);
    list_tmp = []
    list_tmp.append(Subset(dataset, random_list_indx[0:int(len(dataset)/3)]))
    list_tmp.append(Subset(dataset, random_list_indx[int(len(dataset)/3):len(dataset)-1]))
                                                        
    dataloader_train_2 = DataLoader(data.ConcatDataset(list_tmp), batch_size=args.batchsize, shuffle=True);
    
    dataloader_train_3 = DataLoader(Subset(dataset, random_list_indx[0:int(len(dataset)/3)*2]), 
                            batch_size=args.batchsize, shuffle=True);
    toReturn = [
                [dataloader_train_1, dataloader_val_1],  
                [dataloader_train_2, dataloader_val_2], 
                [dataloader_train_3,dataloader_val_3]
                ]
    return toReturn


#%%#############################

if __name__ == '__main__':
    
    #%%  Parse input arguments
    args = parser.parse_args()
    
    #%% Check device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print('Selected device:', device)

    #%% Create dataset
    trans = transforms.Compose([RandomCrop(args.crop_len),
                                OneHotEncoder(args.alphabet_len),
                                ToTensor()
                                ])
    
    dataset = EarnestDataset(filepath=args.datasetpath, crop_len=args.crop_len, transform=trans)
    
    #%% Initialize network
    net = Network(input_size=args.alphabet_len, 
                  hidden_units=args.hidden_units, 
                  layers_num=args.layers_num, 
                  dropout_prob=args.dropout_prob)
    net.to(device)

 
    #%% FInal network training
    hidden_units = 256
    crop_len = 40
    lrate = 0.001
    
    trans = transforms.Compose([RandomCrop(crop_len),
                                OneHotEncoder(args.alphabet_len),
                                ToTensor()
                                ])
    
    dataset = EarnestDataset(filepath=args.datasetpath, crop_len=crop_len, transform=trans)
    
    
    dataloader = DataLoader(dataset, batch_size=args.batchsize, shuffle=True);

    
    net = Network(input_size=args.alphabet_len, 
                                  hidden_units=hidden_units, 
                                  layers_num=args.layers_num, 
                                  dropout_prob=args.dropout_prob)
    net.to(device)
    optimizer = torch.optim.RMSprop(net.parameters(), lr = lrate)
    loss_fn = nn.CrossEntropyLoss();

    memory_loss = []
    memory_min_temp = 100
    final_guard = True
    counter_final_guard = 0
    for epoch in range(args.num_epochs):
        if (epoch + 1) % 100 == 0:
            print('##################################')
            print('## EPOCH %d' % (epoch + 1))
            print('##################################')
        if(final_guard):
            # Iterate batches
            for batch_sample in dataloader:
                # Extract batch
                batch_onehot = batch_sample['encoded_onehot'].to(device);
                # Update network
                batch_loss =  train_batch(net, batch_onehot, loss_fn, optimizer);
                
                memory_loss.append(batch_loss)

                
                if (epoch > 10000):
                    if(memory_min_temp < batch_loss):
                        counter_final_guard = counter_final_guard + 1
                        if (counter_final_guard == 200):
                            ### Save all needed parameters
                            # Create output dir
                            out_dir = Path("Final results - epochs #" + str(epoch +1));
                            out_dir.mkdir(parents=True, exist_ok=True);
                            # Save network parameters
                            torch.save(net.state_dict(), out_dir / 'net_params.pth');
                            # Save training parameters
                            with open(out_dir / 'training_args.json', 'w') as f:
                                json.dump(vars(args), f, indent=4)
                            # Save encoder dictionary
                            with open(out_dir / 'char_to_number.json', 'w') as f:
                                json.dump(dataset.char_to_number, f, indent=4)
                            # Save decoder dictionary
                            with open(out_dir / 'number_to_char.json', 'w') as f:
                                json.dump(dataset.number_to_char, f, indent=4)
                            final_guard = False
                            break
                    else:
                        memory_min_temp = batch_loss
                        counter_final_guard = 0
                
                if (epoch + 1) % 100 == 0:
                    print('\t Training loss (single batch):', batch_loss)
    
            if (epoch + 1) % 10000 == 0:
                ### Save all needed parameters
                # Create output dir
                out_dir = Path("Richard3_post_Kfold " + str(epoch +1));
                out_dir.mkdir(parents=True, exist_ok=True);
                # Save network parameters
                torch.save(net.state_dict(), out_dir / 'net_params.pth');
                # Save training parameters
                with open(out_dir / 'training_args.json', 'w') as f:
                    json.dump(vars(args), f, indent=4)
                # Save encoder dictionary
                with open(out_dir / 'char_to_number.json', 'w') as f:
                    json.dump(dataset.char_to_number, f, indent=4)
                # Save decoder dictionary
                with open(out_dir / 'number_to_char.json', 'w') as f:
                    json.dump(dataset.number_to_char, f, indent=4)
        else:
            break
    
 
