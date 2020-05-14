# -*- coding: utf-8 -*-

import json
import torch
from torch import nn
import argparse
from network import Network
from richard3_dataset import encode_text, create_one_hot_matrix
from pathlib import Path
import numpy as np
## PARAMETERS
parser = argparse.ArgumentParser(description='Generate a chapter starting from a given text')

parser.add_argument('--seed', type=str, default='blood of my enemies', help='Initial text of the chapter')
parser.add_argument('--model_dir',  type=str, default='Final results - epochs #10673', help='Network model directory')
parser.add_argument('--length', type=int, default=536, help='Length')


# an implementation of softmax with temperature to sample chars

def char_sampler(input_to_soft, temperature):
    softmax = nn.functional.softmax(input_to_soft, dim=1)
    EPSILON = 10e-16 # to avoid taking the log of zero
    #print(preds)
    (np.array(softmax) + EPSILON).astype('float64')
    preds = torch.log(softmax) / temperature
    #print(preds)
    exp_preds = torch.exp(preds)
    #print(exp_preds)
    preds = exp_preds / torch.sum(exp_preds)
    #print(preds)
    probas = torch.multinomial(preds, 1)
    return probas.item()

#%%
if __name__ == '__main__':
 #%%   
    ### Parse input arguments
    args = parser.parse_args()
    max_length = args.length
    #%% Load training parameters
    model_dir = Path(args.model_dir)
    print ('Loading model from: %s' % model_dir)
    training_args = json.load(open(model_dir / 'training_args.json'))
      
    #%% Load encoder and decoder dictionaries
    number_to_char = json.load(open(model_dir / 'number_to_char.json'))
    char_to_number = json.load(open(model_dir / 'char_to_number.json'))
        
    #%% Initialize network
    net = Network(input_size=training_args['alphabet_len'], 
                  hidden_units=training_args['hidden_units'], 
                  layers_num=training_args['layers_num'])
        
    #%% Load network trained parameters
    net.load_state_dict(torch.load(model_dir / 'net_params.pth', map_location='cpu'))
    net.eval() # Evaluation mode (e.g. disable dropout)
    
    # choose the best temperatures manually
    temp_value =0.15
    

    #%% Find initial state of the RNN
    
    print("\n###")
    
    with torch.no_grad():
        # Encode seed
        seed_encoded = encode_text(char_to_number, args.seed)
        # One hot matrix
        seed_onehot = create_one_hot_matrix(seed_encoded, training_args['alphabet_len'])
        # To tensor
        seed_onehot = torch.tensor(seed_onehot).float()
        # Add batch axis
        seed_onehot = seed_onehot.unsqueeze(0)
        # Forward pass
        net_out, net_state = net(seed_onehot)
        # Get the most probable last output index
        next_char_encoded = char_sampler(net_out[:, -1, :],temp_value)
        # Print the seed letters
        print(args.seed, end='', flush=True)
        next_char = number_to_char[str(next_char_encoded)]
        print(next_char, end='', flush=True)
        
    #%% Generate chapter
    new_line_count = 1
    tot_char_count = 0
    while tot_char_count < max_length:
        with torch.no_grad(): # No need to track the gradients
            # The new network input is the one hot encoding of the last chosen letter
            net_input = create_one_hot_matrix([next_char_encoded], training_args['alphabet_len'])
            net_input = torch.tensor(net_input).float()
            net_input = net_input.unsqueeze(0)
            # Forward pass
            net_out, net_state = net(net_input, net_state)
            # Get the most probable letter index
            
            next_char_encoded = char_sampler(net_out[:, -1, :],temp_value)
            # Decode the letter
            next_char = number_to_char[str(next_char_encoded)]
            print(next_char, end='', flush=True)
            # Count total letters
            tot_char_count += 1
            # Count new lines
            if next_char == '\n':
                new_line_count += 1
    print("\n###")
    print("\n***\nPrinted chars: " + str(tot_char_count) + "\n" + "Printed lines: " + str(new_line_count) + "\n***")
        
        
        