import os
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import random

from models import ANNModel
from data_loaders import load_monkey_I, load_monkey_M

SEEDS = [1, 11, 111]


def get_file_names(folder_path, monkey):
    file_names = []
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path):
            file_names.append(item)
    if monkey == "I":
        file_names = sorted(file_names, key=lambda x: x.split('_')[1])
    elif monkey == "M":
        file_names = sorted(file_names, key=lambda x: x.split('_')[1].split('-')[2])
    return file_names

    
def train_network(net, train_set_loader, epochs = 50, lr = 0.001):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr= lr, momentum=0.9)
    
    for epoch in range(epochs):
        net.train()
        print_loss = 0.0
        train_loss = 0.0
        
        for k, data in enumerate(train_set_loader):

            input, labels= data
            optimizer.zero_grad()
            outputs = net(input)
            outputs = outputs.squeeze()
            labels = labels.squeeze()
            loss = criterion(outputs, labels)
        
            loss.backward()
            optimizer.step()
            
            print_loss += loss.item()
            train_loss += loss.item()
            
            print_num = 200
            if k % print_num == print_num - 1: 
                    print(f'[{epoch + 1}, {k + 1:5d}] loss: {print_loss / print_num:.3f}')
                    print_loss = 0.0
    return net

def test_network(net, test_set_loader):
    r2_res = 0
    net.eval()
    with torch.no_grad():
        all_preds = []
        all_labels = []
        for data in test_set_loader:
            inputs, labels = data
            labels = labels.squeeze()
            preds = net(inputs)
            preds = preds.squeeze()
    
            if preds.dim() > 0 and preds.numel() > 0 and labels.numel() > 0:
                all_preds.append(preds)
                all_labels.append(labels)
        
        # Concatenate all batches
        if len(all_preds) > 0:
            try:
                all_preds = torch.cat(all_preds)
                all_labels = torch.cat(all_labels)
                
                # Calculate R2 directly
                mean_labels = torch.mean(all_labels)
                ss_total = torch.sum((all_labels - mean_labels) ** 2)
                ss_residual = torch.sum((all_labels - all_preds) ** 2)
                r2_res = 1 - (ss_residual / ss_total)
            except Exception as e:
                print(f"Error during concatenation: {e}")
                # If we encounter an error, just return empty results
                r2_res = 0
                all_preds = torch.tensor([])
                all_labels = torch.tensor([])
        else:
            r2_res = 0
            all_preds = torch.tensor([])
            all_labels = torch.tensor([])
    
    return r2_res, all_preds, all_labels

def sequential_learning(file_path, save_folder, monkey = "I"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_idx = 0
    
    files = get_file_names(file_path, monkey=monkey)
    print(files)
    
    num_trials = len(files)
    fall_offs = np.zeros((num_trials,15))
    incrementals = np.zeros((num_trials,15))

    
    for seed_num, seed in enumerate(SEEDS):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        for iteration in range(5):

            file = files[0]

            test1 = []
            test2 = []
            test3 = []
            test4 = []

            trained_full = []

            lengths = []
            spike_counts_list= []
            if monkey == 'I':
                train_loader, test_loader = load_monkey_I(file_path, file, 256, fold = iteration)
            
            elif monkey == 'M':
                train_loader, test_loader = load_monkey_M(file_path, file, 256, fold = iteration)
                

            base_model = ANNModel(input_dim=96, layer1=32, layer2=48, output_dim=1, drop_rate=0.5)
            base_model = train_network(base_model, train_loader, epochs = 25)
            torch.save(base_model.state_dict(), save_folder + '/iteration' + str(iteration) + '_seed' + str(seed_num) + '_session0.pth')

            state_dict = torch.load(save_folder + '/iteration' + str(iteration) + '_seed' + str(seed_num) + '_session0.pth', map_location=device, weights_only=True)
            base_model.load_state_dict(state_dict)

            sequential = ANNModel(input_dim=96, layer1=32, layer2=48, output_dim=1, drop_rate=0.5)
            sequential.load_state_dict(state_dict)


            for file_num, file in enumerate(files):
               
                if monkey == 'I':
                    train_loader, test_loader = load_monkey_I(file_path, file, 256, fold = iteration)
                
                elif monkey == 'M':
                    train_loader, test_loader = load_monkey_M(file_path, file, 256, fold = iteration)
                
                
                # #Test 1: Train new network for 1 Epoch
                one_epoch = ANNModel(input_dim=96, layer1=32, layer2=48, output_dim=1, drop_rate=0.5)
                temp = train_network(one_epoch, train_loader, epochs = 1)
                r2_res,  all_preds, all_labels = test_network(temp, test_loader)
                test1.append(r2_res)
                
                # #Test 2: Model trained for 50 epochs on session 0, then 1 epoch on each
                non_sequential = ANNModel(input_dim=96, layer1=32, layer2=48, output_dim=1, drop_rate=0.5)
                state_dict = torch.load(save_folder + '/iteration' + str(iteration) + '_seed' + str(seed_num) + '_session0.pth', map_location=device, weights_only=True)
                non_sequential.load_state_dict(state_dict)
                if file_num != 0:
                    temp2 = train_network(non_sequential, train_loader, epochs = 1)
                    r2_res,  all_preds, all_labels = test_network(temp2, test_loader)
                else:
                    r2_res,  all_preds, all_labels = test_network(non_sequential, test_loader)
                test2.append(r2_res)
                    

                if file_num != 0:
                    sequential = train_network(sequential, train_loader, epochs = 1)
                    r2_res,  all_preds, all_labels = test_network(sequential, test_loader)
                    test3.append(r2_res)
                    torch.save(sequential.state_dict(), save_folder + '/iteration' + str(iteration) + '_seed' + str(seed_num) + '_session' +str(file_num) + '.pth')
                else:
                    r2_res,  all_preds, all_labels = test_network(sequential, test_loader)
                    test3.append(r2_res)
                
                #Test 4: Only trained on session 0
                r2_res,  all_preds, all_labels = test_network(base_model, test_loader)
                test4.append(r2_res)

            incrementals[:, run_idx] = np.array(test3)
            fall_offs[:, run_idx] = np.array(test4)
            
            run_idx += 1
    return incrementals, fall_offs
    
    