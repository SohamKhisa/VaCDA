import numpy as np
import sys
import os
import torch
sys.path.append(os.getcwd() + '/../util')
from model import Classifier, Encoder
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

from miscellaneous import reparameterize
from visualization import plot_confusion_matrix

def testVaCDA(save_path, task, activities, domains, dataset, latent_dim, target, encoder_init_channel, encoder_out_channel, T_test, DEVICE):
    print("\n-----------------------------------------")
    print("------------- VaCDA TESTING -------------")
    print("-----------------------------------------")
    #===================================
    # Load Classifier
    #===================================
    output_gt_number = len(activities)
    classifier = Classifier(output_gt_number, infeature=latent_dim).to(DEVICE)
    classifier_load_path = save_path + "classifier.pth"
    classifier.load_state_dict(torch.load(classifier_load_path, weights_only=True))

    #===================================
    # Logging
    #===================================
    test_filename = save_path + 'result.txt'
    test_file = open(test_filename, "w")

    #===================================
    # Load Encoder
    #===================================
    encoder = Encoder(latent_dim, init_channel=encoder_init_channel, last_channel=encoder_out_channel).to(DEVICE)
    encoder_load_path = save_path + "encoder.pth"
    encoder.load_state_dict(torch.load(encoder_load_path, weights_only=True))

    #====================================
    # Classifier Testing
    #====================================
    classifier.eval()
    encoder.eval()
    with torch.no_grad():
        task_correct = 0
        task_total = 0
        all_preds = []
        all_labels = []
        
        for vidx, (_, testSample, testLabel) in enumerate(T_test):
            testSample, testLabel = testSample.to(DEVICE).float(), testLabel.to(DEVICE).long()
            # Get mean and logvar
            mu, logvar = encoder(testSample)

            # Reparameterization trick
            z, _, _ = reparameterize(mu, logvar)

            # outputs
            pred = classifier(z)

            _, task_pred = torch.max(pred.data, 1)
            all_preds.extend(task_pred.cpu().numpy())
            all_labels.extend(testLabel.cpu().numpy())

            task_correct += (task_pred == testLabel).sum()
            task_total += pred.size(0)
            
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        testAcc = float(task_correct) * 100 / task_total
        precision = precision_score(all_labels, all_preds, average='micro')
        recall = recall_score(all_labels, all_preds, average='micro')
        f1 = f1_score(all_labels, all_preds, average='micro')
        
        if task == "position":
            target_string = "_".join([domains[i] for i in target])
            print(f"Dataset: {dataset}, all persons")
            print(f"Test_accuracy_target-{target_string}: {testAcc}")
            print(f"Test_precision_target-{target_string}: {precision}")
            print(f"Test_recall_target-{target_string}: {recall}")
            print(f"Test_f1_target-{target_string}: {f1}\n")

            test_file.write(f"Dataset: {dataset}, all persons\n")
            test_file.write(f"Test_accuracy_target-{target_string}:{testAcc}\n")
            test_file.write(f"Test_precision_target-{target_string}:{precision}\n")
            test_file.write(f"Test_recall_target-{target_string}:{recall}\n")
            test_file.write(f"Test_f1_target-{target_string}:{f1}\n")
            
        elif task == "person":
            target_string = ", ".join([domains[i] for i in target])
            print(f"Dataset: {dataset}, all Positions")
            print(f"Test_accuracy_target-({target_string}): {testAcc}")
            print(f"Test_precision_target-({target_string}): {precision}")
            print(f"Test_recall_target-({target_string}): {recall}")
            print(f"Test_f1_target-({target_string}): {f1}\n")

            test_file.write(f"Dataset: {dataset}, all positions\n")
            test_file.write(f"Test_accuracy_target-({target_string}):{testAcc}\n")
            test_file.write(f"Test_precision_target-({target_string}):{precision}\n")
            test_file.write(f"Test_recall_target-({target_string}):{recall}\n")
            test_file.write(f"Test_f1_target-({target_string}):{f1}\n")
        
        elif task == "device":
            target_string = ", ".join([domains[i] for i in target])
            print(f"Dataset: {dataset}, all persons")
            print(f"Test_accuracy_target-({target_string}): {testAcc}")
            print(f"Test_precision_target-({target_string}): {precision}")
            print(f"Test_recall_target-({target_string}): {recall}")
            print(f"Test_f1_target-({target_string}): {f1}\n")

            test_file.write(f"Dataset: {dataset}, all persons\n")
            test_file.write(f"Test_accuracy_target-({target_string}):{testAcc}\n")
            test_file.write(f"Test_precision_target-({target_string}):{precision}\n")
            test_file.write(f"Test_recall_target-({target_string}):{recall}\n")
            test_file.write(f"Test_f1_target-({target_string}):{f1}\n")
            
        test_file.close()
        # Compute the confusion matrix
        conmat = confusion_matrix(all_labels, all_preds)
        # Plot and save the confusion matrix
        plot_save_path = save_path  # Save path for the confusion matrix plot
        plot_confusion_matrix(conmat, classes=activities, normalize=False, title='Confusion_Matrix', plot_save_path=plot_save_path, show=False)