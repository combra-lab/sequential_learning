import torch
import numpy as np
import os
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from anatome import Distance

from models import ANNModel
from data_loaders import load_monkey_I, load_monkey_M
from utils import get_file_names

def model_similarity(model_folder, data_folder, monkey):
    
    if monkey == "I":
        num_sessions = 12
    elif monkey == "M":
        num_sessions = 9
        
    files = get_file_names(data_folder, monkey)
        
    res_list_opd = np.zeros((num_sessions-1, 15))
    res_list_cka = np.zeros((num_sessions-1, 15))
    
        
    res_lists = [res_list_opd, res_list_cka]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    session_nums = np.arange(1,num_sessions)
    for session_num in session_nums:
        run_idx = 0
        for seed_num in range(3):
            for fold in range(5):
                
                if monkey == "I":
                    train_loader, test_loader = load_monkey_I(file_path= data_folder, filename = files[0], batch_size= 256, fold=fold)
                elif monkey == "M":
                    train_loader, test_loader = load_monkey_M(file_path= data_folder, file = files[0], batch_size= 256, fold=fold)
                    

            
                
                model1 = ANNModel(input_dim=96, layer1=32, layer2=48, output_dim=1, drop_rate=0.5)
                model2 = ANNModel(input_dim=96, layer1=32, layer2=48, output_dim=1, drop_rate=0.5)
                state_dict_1 = torch.load(model_folder + '/iteration' + str(fold) + '_seed' + str(seed_num) + '_session' +str(session_num - 1) + '.pth', map_location=device, weights_only=False)
                state_dict_2 = torch.load(model_folder + '/iteration' + str(fold) + '_seed' + str(seed_num) + '_session' +str(session_num) + '.pth', map_location=device, weights_only=False)
                model1.load_state_dict(state_dict_1)
                model2.load_state_dict(state_dict_2)

                model1.eval()
                model2.eval()
                opd_distance = Distance(model1, model2, method='opd')
                pwcca_distance = Distance(model1, model2, method='pwcca')
                svcca_distance = Distance(model1, model2, method='svcca')
                cka_distance = Distance(model1, model2, method = 'lincka')

                distances = [opd_distance, cka_distance]

                with torch.no_grad():
                    for inputs, labels in test_loader:
                        for distance in distances:
                        #print(inputs.shape)
                            distance.forward(inputs)

                for i, distance in enumerate(distances):
                    layer1_distance = distance.between('fc1', 'fc1')
                    layer2_distance = distance.between('fc2', 'fc2')
                    layer3_distance = distance.between('fc3', 'fc3')
                    mean_distance = (layer1_distance + layer2_distance + layer3_distance) / 3

                    res_lists[i][session_num-1, run_idx] = mean_distance
                    
                run_idx += 1
    return res_lists
    
    

