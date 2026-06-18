import os
import sys
import numpy as np
import random
import torch
import torch.nn.functional as F
import argparse
import pandas as pd
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from model import Encoder, Decoder, ProjectionHead, Classifier
from train import trainVAE, trainCLF
from plot_loss import plot_losses, plot_classifier_metrics
from test import testVaCDA

base_url = os.getcwd()
sys.path.append(base_url + '/../util')

from model import *
from visualization import *
from datapreprocess import *
from miscellaneous import *


os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
def set_seed(seed=42, loader=None):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    try:
        loader.sampler.generator.manual_seed(seed)
    except AttributeError:
        pass


def config_experiment(task, dataset):
    domains, activities, person_list, position_list = None, None, None, None
    device_list, person_list_phone, person_list_watch = None, None, None
    person_map = None
    AXIS, FROM, TO, START, END = None, None, None, None, None

    if task == "position":
        if dataset == "dsads":
            domains = {0:"TORSO", 1:"RA", 2:"LA", 3:"RL", 4:"LL"}
            activities = [ "standing", "lying-back", "ascending", "walking-parking-lot", "treadmill-running", "stepper-exercise", "cross-trainer-exercise", "rowing", "jumping",  "playing-basketball"]
            person_list = ["User1","User2","User3","User4", "User5","User6","User7","User8"]
            position_list = ["TORSO","RA","LA","RL","LL"]
        elif dataset == "opp":
            domains = {0:"BACK", 1:"RUA", 2:"RLA", 3:"LUA", 4:"LLA"}
            activities = ['Sitting','Standing','Walking','Running']
            person_list = ["U1","U2","U3","U4"]
            position_list = ["BACK", "RUA", "RLA", "LUA","LLA"]
        elif dataset == "pamap2":
            domains = {0:"Wrist", 1:"Chest", 2:"Ankle"}
            activities = ['lying', 'sitting', 'standing', 'walking', 'running', 'vacuum', 'ironing', 'rope_jumping']
            person_list = ["U1","U2","U4","U5","U6","U7","U8"]
            position_list = ['Wrist', 'Chest', 'Ankle']
        elif dataset == "wisdm":
            print(f"An experiment has not been configured for the {dataset} dataset.")
            sys.exit(1)
        else:
            print(f"Unknown dataset: {dataset}")
            sys.exit(1)
            
    elif task == "person":
        if dataset == "dsads":
            domains = {0:"User1", 1:"User2", 2:"User3", 3:"User4", 4:"User5", 5:"User6", 6:"User7", 7:"User8"}
            position_list = ["TORSO", "RA", "LA", "RL", "LL"]
            activities = ["standing", "lying-back", "ascending", "walking-parking-lot", "treadmill-running", "stepper-exercise", "cross-trainer-exercise", "rowing", "jumping",  "playing-basketball"]
            person_list = ["User1","User2","User3","User4","User5","User6","User7","User8"]
        elif dataset == "opp":
            domains = {0:"U1", 1:"U2", 2:"U3", 3:"U4"}
            position_list = ["BACK", "RUA", "RLA", "LUA","LLA"]
            activities = ['Sitting','Standing','Walking','Running']
            person_list = ["U1","U2","U3","U4"]
        elif dataset == "pamap2":
            domains = {0:'U1', 1:'U2', 2:'U4', 3:'U5', 4:'U6', 5:'U7', 6:'U8'}
            activities = ['lying', 'sitting', 'standing', 'walking', 'vacuum', 'ironing']
            position_list = ['Wrist', 'Chest', 'Ankle']
            person_list = ["U1","U2","U4","U5","U6","U7","U8"]
        elif dataset == "wisdm":
            domains = {0:"U1", 1:"U5", 2:"U7", 3:"U8", 4:"U12", 5:"U13", 6:"U15", 7:"U37", 8:"U38", 9:"U39", 10:"U40"}
            position_list = ["phone", 'watch']
            activities = ['walking', 'jogging', 'stairs', 'sitting', 'standing', 'kicking', 'catch', 'dribbling']
            person_list = ["U1","U5","U7","U8","U12","U13","U15","U37","U38","U39","U40"]
        else:
            print(f"Unknown dataset: {dataset}")
            sys.exit(1)

    elif task == "device":
        if dataset == "wisdm":
            domains = {0:"phone", 1:"watch"}
            device_list = ["phone", "watch"]
            activities = ['walking', 'jogging', 'stairs', 'sitting', 'standing', 'kicking', 'catch', 'dribbling']
            person_list_phone = ['U1', 'U5', 'U7', 'U8', 'U12', 'U13', 'U15', 'U16', 'U18', 'U23', 'U25', 'U31', 'U32', 'U33', 'U35', 'U37', 'U38', 'U39', 'U40', 'U41', 'U42', 'U43', 'U44', 'U45', 'U46', 'U47', 'U49', 'U51']
            person_list_watch = ['U2', 'U4', 'U5', 'U7', 'U11', 'U12', 'U14', 'U15', 'U16', 'U17', 'U19', 'U20', 'U23', 'U24', 'U28', 'U30', 'U31', 'U32', 'U33', 'U35', 'U37', 'U38', 'U39', 'U40', 'U41', 'U42', 'U44', 'U47', 'U49']
            person_map = {0:person_list_phone, 1:person_list_watch}
        elif dataset=="dsads" or dataset=='pamap2' or dataset=='opp':
            print(f"An experiment has not been configured for the {dataset} dataset.")
            sys.exit(1)
        else:
            print(f"Unknown dataset: {dataset}")
            sys.exit(1)

    if dataset == "opp":
        AXIS = 3
        FROM = 0 # for reading imu data
        TO = FROM+3
        START = 3 # for activity label
        END = 4
    elif dataset == "wisdm":
        AXIS = 3
        FROM = 2 # for reading imu data
        TO = FROM+3
        START = 1 # for activity label
        END = 2
    elif dataset == "pamap2" or dataset == "dsads":
        AXIS = 3
        FROM = 0 # for reading imu data
        TO = FROM+3
        START = 4 # for activity label
        END = 5
    else:
        print(f"Unknown dataset: {dataset}")
        sys.exit(1)

    return domains, activities, person_list, position_list, device_list, person_map, AXIS, FROM, TO, START, END



def config_io(task, dataset, domains, target, activities):
    activity_num = len(activities)
    if task == "position":
        target_string = "_".join([domains[i] for i in target])
        if dataset == "dsads":
            folder_name = str(activity_num)+ "_activity"+"_window_"+str(win_size)+ "_overlap_"+str(overlap)
            dataset_path = base_url + "/../dataset_preprocess/dsads/data_files/"+folder_name+"/"
            save_path = base_url + "/output-v211/dsads-cross-position-heterogeneity/" + "target-position-" + target_string + '-seed' + str(seed) + "/"
        elif dataset == "opp":
            folder_name = str(activity_num)+ "_activity"+"_window_"+str(win_size)+ "_overlap_"+str(overlap)
            dataset_path = base_url + "/../dataset_preprocess/opportunity/Data Files/"+folder_name+"/"
            save_path = base_url + "/output-v211/opp-cross-position-heterogeneity/" + "target-position-" + target_string + '-seed' + str(seed) + "/"
        elif dataset == "pamap2":
            folder_name = str(activity_num)+ "_activity"+"_window_"+str(win_size)+ "_overlap_"+str(overlap)
            dataset_path = base_url + "/../dataset_preprocess/pamap2/data_files/"+folder_name+"/"
            save_path = base_url + "/output-v211/pamap2-cross-position-heterogeneity/" + "target-position-" + target_string + '-seed' + str(seed) + "/"
            
    elif task == "person":
        target_string = "_".join([domains[i] for i in target])
        if dataset == "dsads":
            folder_name = str(activity_num)+ "_activity"+"_window_"+str(win_size)+ "_overlap_"+str(overlap)
            dataset_path = base_url + "/../dataset_preprocess/dsads/data_files/"+folder_name+"/"
            save_path = base_url + "/output-v211/dsads-cross-person-heterogeneity/" + "target-person-" + target_string + '-seed' + str(seed) + "/"
        elif dataset == "opp":
            folder_name = str(activity_num)+ "_activity"+"_window_"+str(win_size)+ "_overlap_"+str(overlap)
            dataset_path = base_url + "/../dataset_preprocess/opportunity/data_files/"+folder_name+"/"
            save_path = base_url + "/output-v211/opp-cross-person-heterogeneity/" + "target-person-" + target_string + '-seed' + str(seed) + "/"
        elif dataset == "pamap2":
            folder_name = str(activity_num)+ "_Activity"+"_Window_"+str(win_size)+ "_Overlap_"+str(overlap)
            dataset_path = base_url + "/../dataset_preprocess/pamap2/data_files/"+folder_name+"/"
            save_path = base_url + "/output-v211/pamap2-cross-person-heterogeneity/" + "target-person-" + target_string + '-seed' + str(seed) + "/"
        elif dataset == "wisdm":
            target_string = "_".join([domains[i] for i in target])
            folder_name = str(activity_num)+ "_activity"+"_window_"+str(win_size)+ "_overlap_"+str(overlap)
            dataset_path = base_url + "/../data_preprocess/wisdm/data_files/" + folder_name + "/"
            save_path = base_url + "/output/wisdm-cross-person-heterogeneity/" + "target-person" + target_string + '-seed' + str(seed) + "/"

    elif task == "device":
        target_string = "_".join([domains[i] for i in target])
        if dataset == "wisdm":
            folder_name = str(activity_num)+ "_activity"+"_window_"+str(win_size)+ "_overlap_"+str(overlap)
            dataset_path = base_url + "/../dataset_preprocess/wisdm/data_files/" + folder_name + "/"
            save_path = base_url + "/output-v211/wisdm-cross-device-heterogeneity/" + "target-device-" + target_string + '-seed' + str(seed) + "/"
            
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    return save_path, dataset_path




def list2numpy(s_train, s_gt_train, s_valid, s_gt_valid, s_test, s_gt_test, t_train, t_gt_train, t_valid, t_gt_valid, t_test, t_gt_test):
    # Converting sources (lists) to list of numpy arrays. s_train[0] will contain the numpy array of the 1st source domain
    for i, row in enumerate(s_train):
        flattened_row = np.concatenate(row).astype(np.float32)
        # print(f"Shape of row {i}: {flattened_row.shape}")
        s_train[i] = flattened_row
    s_gt_train = [np.array(sublist).astype(np.float32) for sublist in s_gt_train]

    #=====================================================================
    # source valid
    for i, row in enumerate(s_valid):
        flattened_row = np.concatenate(row).astype(np.float32)
        # print(f"Shape of row {i}: {flattened_row.shape}")
        s_valid[i] = flattened_row
    s_gt_valid = [np.array(sublist).astype(np.float32) for sublist in s_gt_valid]

    # source test
    for i, row in enumerate(s_test):
        flattened_row = np.concatenate(row).astype(np.float32)
        # print(f"Shape of row {i}: {flattened_row.shape}")
        s_test[i] = flattened_row
    s_gt_test = [np.array(sublist).astype(np.float32) for sublist in s_gt_test]

    #Target:
    t_train = np.concatenate(t_train, axis=0).astype(np.float32)
    t_gt_train = np.array(t_gt_train).astype(np.float32)
    t_valid = np.concatenate(t_valid, axis=0).astype(np.float32)
    t_gt_valid = np.array(t_gt_valid).astype(np.float32)
    t_test = np.concatenate(t_test, axis=0).astype(np.float32)
    t_gt_test = np.array(t_gt_test).astype(np.float32)

    return s_train, s_gt_train, s_valid, s_gt_valid, s_test, s_gt_test, t_train, t_gt_train, t_valid, t_gt_valid, t_test, t_gt_test





def load_data(task, sources, dataset_path, item, position_array, person_array, device_list, person_map, win_size, batch_size, overlap, FROM, TO, START, END, AXIS):
    # sources can be [0, 1, 3, 4], and target = 2
    step_size = int(win_size * (1 - overlap))
    nsources = len(sources)
    #===================================================#
    s_train = [[] for _ in range(nsources)]
    t_train = []

    s_gt_train = [[] for _ in range(nsources)]
    t_gt_train = []
    #===================================================#
    s_valid = [[] for _ in range(nsources)]
    t_valid = []

    s_gt_valid = [[] for _ in range(nsources)]
    t_gt_valid = []
    #===================================================#
    s_test = [[] for _ in range(nsources)]
    t_test = []

    s_gt_test = [[] for _ in range(nsources)]
    t_gt_test = []
    #===================================================#
    if task == 'position':
        print('Source')
        for i in range(len(sources)):
            # Create an empty DataFrame to hold the aggregated data for this source position
            df_aggregated_train = pd.DataFrame()
            df_aggregated_valid = pd.DataFrame()
            df_aggregated_test = pd.DataFrame()

            # Combine data from all users for the current source position
            for person in person_list:
                for split_index in range(0, 3):
                    file_name = person + "_" + position_array[sources[i]] + '_' + item[split_index]
                    print(f"Processing: {file_name}")
                    df = pd.read_csv(dataset_path + file_name + '.csv', sep=",")

                    # Aggregate data into the corresponding split DataFrame
                    if split_index == 0:
                        df_aggregated_train = pd.concat([df_aggregated_train, df], ignore_index=True)
                    elif split_index == 1:
                        df_aggregated_valid = pd.concat([df_aggregated_valid, df], ignore_index=True)
                    elif split_index == 2:
                        df_aggregated_test = pd.concat([df_aggregated_test, df], ignore_index=True)

            # Pass the aggregated data to calculate_window
            calculate_window(df_aggregated_train, s_train[i], s_gt_train[i], win_size, step_size, FROM, TO, START, END, AXIS)
            calculate_window(df_aggregated_valid, s_valid[i], s_gt_valid[i], win_size, step_size, FROM, TO, START, END, AXIS)
            calculate_window(df_aggregated_test, s_test[i], s_gt_test[i], win_size, step_size, FROM, TO, START, END, AXIS)

        # For the target position, similarly aggregate data across all users
        df_aggregated_train_target = pd.DataFrame()
        df_aggregated_valid_target = pd.DataFrame()
        df_aggregated_test_target = pd.DataFrame()

        print('Target')
        for i in range(len(target)):
            for person in person_list:
                for split_index in range(0, 3):
                    file_name = person + "_" + position_array[target[i]] + '_' + item[split_index]
                    print(f"Processing: {file_name}")
                    df = pd.read_csv(dataset_path + file_name + '.csv', sep=",")

                    if split_index == 0:
                        df_aggregated_train_target = pd.concat([df_aggregated_train_target, df], ignore_index=True)
                    elif split_index == 1:
                        df_aggregated_valid_target = pd.concat([df_aggregated_valid_target, df], ignore_index=True)
                    elif split_index == 2:
                        df_aggregated_test_target = pd.concat([df_aggregated_test_target, df], ignore_index=True)

        calculate_window(df_aggregated_train_target, t_train, t_gt_train, win_size, step_size, FROM, TO, START, END, AXIS)
        calculate_window(df_aggregated_valid_target, t_valid, t_gt_valid, win_size, step_size, FROM, TO, START, END, AXIS)
        calculate_window(df_aggregated_test_target, t_test, t_gt_test, win_size, step_size, FROM, TO, START, END, AXIS)
            
    elif task == 'person':
        print('Source')
        for i in range(len(sources)):
            # Create an empty DataFrame to hold the aggregated data for this source position
            df_aggregated_train = pd.DataFrame()
            df_aggregated_valid = pd.DataFrame()
            df_aggregated_test = pd.DataFrame()

            for position in position_list:
                for split_index in range(0, 3):
                    file_name = person_array[sources[i]] + "_" + position + '_' + item[split_index]
                    print(f"Processing: {file_name}")
                    df = pd.read_csv(dataset_path + file_name + '.csv', sep=",")

                    # Aggregate data into the corresponding split DataFrame
                    if split_index == 0:
                        df_aggregated_train = pd.concat([df_aggregated_train, df], ignore_index=True)
                    elif split_index == 1:
                        df_aggregated_valid = pd.concat([df_aggregated_valid, df], ignore_index=True)
                    elif split_index == 2:
                        df_aggregated_test = pd.concat([df_aggregated_test, df], ignore_index=True)

            # Pass the aggregated data to calculate_window
            calculate_window(df_aggregated_train, s_train[i], s_gt_train[i], win_size, step_size, FROM, TO, START, END, AXIS)
            calculate_window(df_aggregated_valid, s_valid[i], s_gt_valid[i], win_size, step_size, FROM, TO, START, END, AXIS)
            calculate_window(df_aggregated_test, s_test[i], s_gt_test[i], win_size, step_size, FROM, TO, START, END, AXIS)

        # For the target position, similarly aggregate data across all users
        print('Target')
        df_aggregated_train_target = pd.DataFrame()
        df_aggregated_valid_target = pd.DataFrame()
        df_aggregated_test_target = pd.DataFrame()
        for i in range(len(target)):
            for position in position_list:
                for split_index in range(0, 3):
                    file_name = person_array[target[i]] + "_" + position + '_' + item[split_index]
                    print(f"Processing: {file_name}")
                    df = pd.read_csv(dataset_path + file_name + '.csv', sep=",")

                    if split_index == 0:
                        df_aggregated_train_target = pd.concat([df_aggregated_train_target, df], ignore_index=True)
                    elif split_index == 1:
                        df_aggregated_valid_target = pd.concat([df_aggregated_valid_target, df], ignore_index=True)
                    elif split_index == 2:
                        df_aggregated_test_target = pd.concat([df_aggregated_test_target, df], ignore_index=True)

        calculate_window(df_aggregated_train_target, t_train, t_gt_train, win_size, step_size, FROM, TO, START, END, AXIS)
        calculate_window(df_aggregated_valid_target, t_valid, t_gt_valid, win_size, step_size, FROM, TO, START, END, AXIS)
        calculate_window(df_aggregated_test_target, t_test, t_gt_test, win_size, step_size, FROM, TO, START, END, AXIS)

    elif task == "device" and dataset == "wisdm":
        print('Source')
        for i in range(len(sources)):
            # Create an empty DataFrame to hold the aggregated data for this source position
            df_aggregated_train = pd.DataFrame()
            df_aggregated_valid = pd.DataFrame()
            df_aggregated_test = pd.DataFrame()

            for person in person_map[sources[0]]:
            #for person in person_list:
                for split_index in range(0, 3):
                    file_name = person + "_" + device_list[sources[i]] + '_' + item[split_index]
                    print(f"Processing: {file_name}")
                    df = pd.read_csv(dataset_path + file_name + '.csv', sep=",")

                    # Aggregate data into the corresponding split DataFrame
                    if split_index == 0:
                        df_aggregated_train = pd.concat([df_aggregated_train, df], ignore_index=True)
                    elif split_index == 1:
                        df_aggregated_valid = pd.concat([df_aggregated_valid, df], ignore_index=True)
                    elif split_index == 2:
                        df_aggregated_test = pd.concat([df_aggregated_test, df], ignore_index=True)
                        
            # Pass the aggregated data to calculate_window
            calculate_window(df_aggregated_train, s_train[i], s_gt_train[i], win_size, step_size, FROM, TO, START, END, AXIS)
            calculate_window(df_aggregated_valid, s_valid[i], s_gt_valid[i], win_size, step_size, FROM, TO, START, END, AXIS)
            calculate_window(df_aggregated_test, s_test[i], s_gt_test[i], win_size, step_size, FROM, TO, START, END, AXIS)

        # For the target position, similarly aggregate data across all users
        print('Target')
        df_aggregated_train_target = pd.DataFrame()
        df_aggregated_valid_target = pd.DataFrame()
        df_aggregated_test_target = pd.DataFrame()
        
        for i in range(len(target)):
            for person in person_map[target[0]]:
                for split_index in range(0, 3):
                    file_name = person + "_" + device_list[target[i]] + '_' + item[split_index]
                    print(f"Processing: {file_name}")
                    df = pd.read_csv(dataset_path + file_name + '.csv', sep=",")

                    # Aggregate data into the corresponding split DataFrame
                    if split_index == 0:
                        df_aggregated_train_target = pd.concat([df_aggregated_train_target, df], ignore_index=True)
                    elif split_index == 1:
                        df_aggregated_valid_target = pd.concat([df_aggregated_valid_target, df], ignore_index=True)
                    elif split_index == 2:
                        df_aggregated_test_target = pd.concat([df_aggregated_test_target, df], ignore_index=True)
                        
        # Pass the aggregated data to calculate_window
        calculate_window(df_aggregated_train_target, t_train, t_gt_train, win_size, step_size, FROM, TO, START, END, AXIS)
        calculate_window(df_aggregated_valid_target, t_valid, t_gt_valid, win_size, step_size, FROM, TO, START, END, AXIS)
        calculate_window(df_aggregated_test_target, t_test, t_gt_test, win_size, step_size, FROM, TO, START, END, AXIS)
    
    # convert the list of lists to list of numpy arrays
    s_train, s_gt_train, s_valid, s_gt_valid, s_test, s_gt_test, t_train, t_gt_train, t_valid, t_gt_valid, t_test, t_gt_test = list2numpy(s_train, 
                                    s_gt_train, s_valid, s_gt_valid, s_test, s_gt_test, t_train, t_gt_train, t_valid, t_gt_valid, t_test, t_gt_test)
    # Convertin data to Torch Dataloader
    # drop_last = True if you use simclr loss as contrastive loss
    drop_last = False
    T_train = []
    T_valid = []
    T_test = []
    S_train = [[] for _ in range(nsources)]
    S_valid = [[] for _ in range(nsources)]
    S_test = [[] for _ in range(nsources)]

    for i in range(nsources):
        S_train[i], S_valid[i], S_test[i] = modified_load_train_valid_test(s_train[i], s_gt_train[i], s_valid[i], s_gt_valid[i], s_test[i], s_gt_test[i], batch_size, win_size, drop_last)
    T_train, T_valid, T_test = modified_load_train_valid_test(t_train, t_gt_train, t_valid, t_gt_valid, t_test, t_gt_test, batch_size, win_size, drop_last)
        
    return S_train, S_valid, S_test, T_train, T_valid, T_test




def init_modelVAE(DEVICE, encoder_init_channel, decoder_init_channel, encoder_out_channel, latent_dim, projection_hid, projection_dim, LEARNING_RATE, LEARNING_RATE_PROJ, BETA1, BETA2):
    # batch_size means batch size per domain
    # =========================
    # Initialize Network
    # =========================
    encoder = Encoder(latent_dim=latent_dim, init_channel=encoder_init_channel, last_channel=encoder_out_channel).to(DEVICE)
    decoder = Decoder(latent_dim=latent_dim, init_channel=decoder_init_channel, last_channel=decoder_out_channel).to(DEVICE)
    projector = ProjectionHead(in_features=latent_dim, hidden_features=projection_hid, out_features=projection_dim).to(DEVICE)
    # shared_encoder = SharedEncoder().to(DEVICE)

    # =========================
    # Initialize Optimizers
    # =========================
    opt_encoder = optim.Adam(params=list(encoder.parameters()), lr=LEARNING_RATE, betas=(BETA1, BETA2), weight_decay=1e-6)
    opt_decoder = optim.Adam(params=list(decoder.parameters()), lr=LEARNING_RATE, betas=(BETA1, BETA2), weight_decay=1e-6)
    opt_projector = optim.Adam(params=list(projector.parameters()), lr=LEARNING_RATE_PROJ, betas=(BETA1, BETA2), weight_decay=1e-6)

    # =========================
    # Setting up Scheduler
    # =========================
    lr_drop_step_vae = 5 # reduce the LR every lr_drop_step epochs.
    lr_drop_epoch_vae = 10 # keep LR constant for this number of epochs.
    def lr_lambda_vae(epoch):
        if epoch < lr_drop_epoch_vae:
            return 1.0  # Keep the learning rate constant
        else:
            return 0.1 ** ((epoch - lr_drop_epoch_vae) // lr_drop_step_vae)  # Reduce learning rate every `step_size` epochs after `n` epochs

    scheduler_encoder = lr_scheduler.LambdaLR(opt_encoder, lr_lambda=lr_lambda_vae)
    scheduler_decoder = lr_scheduler.LambdaLR(opt_decoder, lr_lambda=lr_lambda_vae)
    scheduler_projector = lr_scheduler.LambdaLR(opt_projector, lr_lambda=lr_lambda_vae)

    return encoder, decoder, projector, opt_encoder, opt_decoder, opt_projector, scheduler_encoder, scheduler_decoder, scheduler_projector


def init_modelCLF(save_path, activities, clf_lr, beta1, beta2, encoder_init_channel, DEVICE):
    #----------------------------------------
    # Initializing Models
    #----------------------------------------
    output_gt_number = len(activities)
    encoder_load_path = save_path + "encoder.pth"
    if not os.path.exists(encoder_load_path):
        print(f"The pretrained encoder {encoder_load_path} does not exist! Please train the encoder first with VAE")
        sys.exit(1)
    encoder = Encoder(latent_dim, init_channel=encoder_init_channel, last_channel=encoder_out_channel).to(DEVICE)
    encoder.load_state_dict(torch.load(encoder_load_path, weights_only=True))

    #----------------------------------------
    # Optimizer and Scheduler
    #----------------------------------------
    classifier = Classifier(output_gt_number, infeature=latent_dim).to(DEVICE)
    optimizer = optim.Adam(params=classifier.parameters(), lr=clf_lr, betas=(beta1, beta2))

    lr_drop_step_clf = 5 # reduce the LR every lr_drop_step epochs. 5
    lr_drop_epoch_clf = 10 # keep LR constant for this number of epochs. 5
    def lr_lambda_clf(epoch):
        if epoch < lr_drop_epoch_clf:
            return 1.0  # Keep the learning rate constant
        else:
            return 0.1 ** ((epoch - lr_drop_epoch_clf) // lr_drop_step_clf)  # Reduce learning rate every `step_size` epochs after `n` epochs

    scheduler_clf = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda_clf)

    return encoder, classifier, optimizer, scheduler_clf



if __name__ == "__main__":
    # Setting up argument parser
    parser = argparse.ArgumentParser(description='The main training and testing file for the VaCDA')
    
    # Add arguments for command-line input
    parser.add_argument('-e', '--experiment', choices=['position', 'person', 'device'], type=str, required=True, help="Specify the experiment type: position (cross-position), person (cross-person), device (cross-device)")
    parser.add_argument('-m', '--mode', choices=['train-vae', 'train-clf', 'testvacda', 'all'], type=str, required=False, default='all', help='Mode of operation we can individually train each component or test the adaptation')
    parser.add_argument('-d', '--dataset', choices=['opp', 'dsads', 'pamap2', 'wisdm'], type=str, required=True, help="Dataset to train and evaluate")
    parser.add_argument('-w', '--win_size', type=int, default=100, required=False, help='Window size to apply sliding window technique on the data')
    parser.add_argument('-o', '--overlap', type=float, default=0.1, required=False, help='Overlap percentage for the sliding window technique')
    parser.add_argument('-s', '--source', nargs='+', required=True, type=int, help="Selected source domain(s)")
    parser.add_argument('-t', '--target', nargs='+', required=True, type=int, help="Selected target domain(s)")
    parser.add_argument('-g', '--gpu', choices=['cuda', 'cpu'], default='cpu', type=str, required=False, help='Specify the device to run the experiment options: cuda, gpu')
    parser.add_argument('-sd', '--seed', type=int, default=42, required=False, help='Seed value for reproducibility')
    parser.add_argument('-ne', "--nepoch", type=int, default=20, required=False, help='Number of epochs for training VAE')
    parser.add_argument('-nc', "--clf_epoch", type=int, default=20, required=False, help='Number of epochs for classifier training')
    parser.add_argument('-b', "--batch_size", type=int, default=32, required=False, help='batch size is not the full batch size. It is the batch size per domain.')
    parser.add_argument('-lr_v', "--learning_rate_vae", type=float, default=0.0001, required=False, help='Learning rate for the VAE training')
    parser.add_argument('-lr_p', "--learning_rate_proj", type=float, default=0.001, required=False, help='Learning rate for projection layers')
    parser.add_argument('-lr_c', "--learning_rate_classifier", type=float, default=0.001, required=False, help='Learning rate for the Classifier')
    parser.add_argument('-m1', "--momentum1", type=float, default=0.9, required=False, help='Momentum factor for the optimizer, typically between 0 and 1 (e.g., 0.9), which accelerates convergence by incorporating past gradients.')
    parser.add_argument('-m2', "--momentum2", type=float, default=0.99, required=False, help='Momentum factor for the squared gradients, typically between 0 and 1 (e.g., 0.999), which helps smooth the optimization by considering past gradient variances.')
    parser.add_argument('-pd', "--projection_dim", type=int, default=128, required=False, help='Projection Dimension for contrastive learning')
    parser.add_argument('-hd', "--projection_hid", type=int, default=2048, required=False, help='Hidden dimension of the projector')
    parser.add_argument('-rw', "--reconstruct_weight", type=int, default=1e6, required=False, help='Reconstruction Weight of for the VAE training')

    # ===========================================================
    # ====================== CONFIGURATION ======================
    # ===========================================================
    # assigning values to the arguments
    args = parser.parse_args()
    task = str(args.experiment)
    dataset = str(args.dataset)
    win_size=int(args.win_size)
    overlap = float(args.overlap)
    mode = str(args.mode)
    source = args.source
    target = args.target
    seed = int(args.seed)
    batch_size = int(args.batch_size)
    LEARNING_RATE = float(args.learning_rate_vae)
    LEARNING_RATE_PROJ = float(args.learning_rate_proj)
    LEARNING_RATE_CLF = float(args.learning_rate_classifier)
    BETA1 = float(args.momentum1)
    BETA2 = float(args.momentum2)
    N_EPOCH = int(args.nepoch)
    CLF_EPOCH = int(args.clf_epoch)
    projection_dim = int(args.projection_dim)
    projection_hid = int(args.projection_hid)
    recon_weight = int(args.reconstruct_weight)
    DEVICE = str(args.gpu)
    
    if DEVICE == 'cuda':
        if torch.cuda.is_available():
            gpu_id = 0
            DEVICE = 'cuda:' + str(gpu_id)
        else:
            print('GPU is not available. Switching to CPU...')
            DEVICE = 'cpu'

    # set for optimal results
    encoder_init_channel = 3
    encoder_out_channel  = 64
    decoder_init_channel = 64
    decoder_out_channel  = 3
    contrastive_weight = 1e8
    latent_dim = 64
    if dataset == "dsads":
        contrastive_weight = 1e3
    # reproducibility
    set_seed(seed=seed)

    domains, activities, person_list, position_list, device_list, person_map, AXIS, FROM, TO, START, END = config_experiment(task, dataset)
    item = ["train","valid","test"]
    print("*************************************")
    if task == "position":
        print("Cross-Position")
    elif task == "person":
        print("Cross-Person")
    elif task == "device":
        print("Cross-Device")
    print("*************************************")
    print(f"Dataset: {dataset}")
    print(f"Source: {[domains[s] for s in source]}")
    print(f"Target: {[domains[t] for t in target]}\n")

    save_path, dataset_path = config_io(task, dataset, domains, target, activities)
    S_train, S_valid, S_test, T_train, T_valid, T_test = load_data(task, source, dataset_path, item, position_list, person_list, device_list, 
                                                            person_map, win_size, batch_size, overlap, FROM, TO, START, END, AXIS)
    
    # ======================================================================
    # ========================== TRAIN VAE =================================
    # ======================================================================
    if mode == 'train-vae' or mode == 'all':
        # Initialize VAE model
        encoder, decoder, projector, opt_encoder, opt_decoder, opt_projector, scheduler_encoder, scheduler_decoder, scheduler_projector = init_modelVAE(DEVICE,
                                        encoder_init_channel, decoder_init_channel, encoder_out_channel, latent_dim, projection_hid, projection_dim,
                                        LEARNING_RATE, LEARNING_RATE_PROJ, BETA1, BETA2)

        trainVAE(save_path, source, activities, S_train, S_valid, S_test, T_train, T_valid, N_EPOCH, encoder, decoder, projector, opt_encoder, 
                opt_decoder, opt_projector, scheduler_encoder, scheduler_decoder, scheduler_projector, recon_weight, contrastive_weight, DEVICE)

    # ===================
    # PLOT VAE LOSSES
    # ===================
    log_file_path = save_path + 'training_log.txt'
    losses_to_plot = ['Train_loss', 'Valid_loss']
    plot_losses(log_file_path, losses_to_plot, save_path)

    losses_to_plot = ['Train_Contrastive_loss', 'Valid_Contrastive_loss']
    plot_losses(log_file_path, losses_to_plot, save_path)

    losses_to_plot = ['Train_KLD', 'Valid_KLD']
    plot_losses(log_file_path, losses_to_plot, save_path)

    losses_to_plot = ['Train_Contrastive_loss2', 'Valid_Contrastive_loss2']
    plot_losses(log_file_path, losses_to_plot, save_path)

    # =============================================================================
    # ========================== TRAIN CLASSIFIER =================================
    # =============================================================================
    if mode == 'train-clf' or mode == 'all':
        # Initialize Classifier model and load the trained Encoder
        encoder, classifier, optimizer_clf, scheduler_clf = init_modelCLF(save_path, activities, LEARNING_RATE_CLF, BETA1, BETA2, encoder_init_channel, DEVICE)
        trainCLF(save_path, source, encoder, classifier, CLF_EPOCH, S_train, S_valid, S_test, scheduler_clf, optimizer_clf, DEVICE)
    # ========================
    # PLOT CLASSIFIER LOSSES
    # ========================
    log_file_path = save_path + "classifier_log.txt"
    save_dir = save_path
    plot_classifier_metrics(log_file_path, save_dir)

    # =========================================================================================
    # ======================== TESTING DOMAIN ADAPTATION ON TARGET DOMAIN =====================
    # =========================================================================================
    if mode == 'testvacda' or mode == 'all':
        testVaCDA(save_path, task, activities, domains, dataset, latent_dim, target, encoder_init_channel, encoder_out_channel, T_test, DEVICE)






    

