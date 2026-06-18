import torch
import torch.nn as nn
import tqdm
import itertools
import sys
import os

sys.path.append(os.getcwd() + '/../util')
from miscellaneous import reparameterize, nt_xent_loss, activity_contrastive_loss

def trainVAE(save_path, sources, activities, S_train, S_valid, S_test, T_train, T_valid, N_EPOCH, encoder, decoder, projector, opt_encoder, opt_decoder, opt_projector, scheduler_encoder, scheduler_decoder, scheduler_projector, recon_weight, contrastive_weight, DEVICE):
    print("\n----------------------------------------")
    print("------------- VAE TRAINING -------------")
    print("----------------------------------------")
    nsources = len(sources)
    activity_num = len(activities)
    #=================
    # Log File
    #=================
    log_filename = save_path + "training_log.txt"
    encoder_save_path = save_path + "encoder.pth"

    # =========================
    # Iterator Setup
    # =========================
    sampleAug = [0] * nsources
    sample = [0] * nsources
    label = [0] * nsources

    S_train_itr_list = []
    S_valid_itr_list = []
    S_test_itr_list = []
            
    for i in range(0, nsources):
        S_train_itr_list.append(iter(S_train[i]))
        S_valid_itr_list.append(iter(S_valid[i]))
        S_test_itr_list.append(iter(S_test[i]))

    #============================
    # Loss
    #============================
    criterion = nn.MSELoss()
    log_file = open(log_filename, "w")

    #============================
    # Training
    #============================
    # train loop
    for epoch in tqdm.tqdm(range(N_EPOCH), desc='total progress'):
        encoder.train()
        decoder.train()
        # shared_encoder.train()
        trainLoss = 0.0
        traintotal = 0
        kldloss = 0.0
        reconLossRegTotal = 0.0
        reconLossAugTotal = 0.0
        contrasTotal = 0.0
        contrasTotal2 = 0.0
        
        for iter_idx, (aug_t_sample, t_sample, _) in enumerate(tqdm.tqdm(T_train)):
            aug_t_sample, t_sample = aug_t_sample.to(DEVICE).float(), t_sample.to(DEVICE).float()
            
            for i in range(0, nsources):
                try:
                    sampleAug[i], sample[i], label[i] = next(S_train_itr_list[i])
                except StopIteration:
                    S_train_itr = iter(S_train[i])
                    sampleAug[i], sample[i], label[i] = next(S_train_itr) 
                sampleAug[i], sample[i], label[i] = sampleAug[i].to(DEVICE).float(), sample[i].to(DEVICE).float(), label[i].to(DEVICE).long()
            
            batch_sample_list = []
            batch_augsample_list = []
            batch_label_list = []
            
            for i in range(0, nsources):
                batch_sample_list.append(sample[i])
                batch_augsample_list.append(sampleAug[i])
                batch_label_list.append(label[i])
            batch_sample_list.append(t_sample)
            batch_augsample_list.append(aug_t_sample)
            # Since we activity_num number of activities indexed from 0 to activity_num-1.
            # Since we are not allowed to use the target domain labels,
            # we are assigning fake label value=activity_num to the target domain activites.
            batch_label_list.append(torch.full((t_sample.size(0),), activity_num).to(DEVICE).long())
                
            batch_samples = torch.cat(batch_sample_list, dim=0)
            batch_augsamples = torch.cat(batch_augsample_list, dim=0)
            batch_labels = torch.cat(batch_label_list, dim=0)
            
            perm_indices = torch.randperm(batch_samples.size(0))
            
            # Shuffle tensors based on the generated indices
            batch_samples = batch_samples[perm_indices]
            batch_augsamples = batch_augsamples[perm_indices]
            batch_labels = batch_labels[perm_indices]
            
            #==========encoder===============
            mu, logvar = encoder(batch_samples)
            muAug,logvarAug = encoder(batch_augsamples)
            
            #===========sampling=============
            z, mu, std = reparameterize(mu, logvar)
            zAug, muAug, stdAug = reparameterize(muAug, logvarAug)
            
            #===========decoder===============
            reconSample = decoder(z)
            reconSampleAug = decoder(zAug)
            
            # =======reconstruction loss======
            reconLoss = criterion(reconSample, batch_samples)
            reconAugLoss = criterion(reconSampleAug, batch_augsamples)
            
            # ==========KL divergence=========
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            # kldAug = -0.5 * torch.sum(1 + logvarAug - muAug.pow(2) - logvarAug.exp())
            
            # ========contrastive loss========
            # ------ contrastive loss 1 ------
            z_projected = projector(z)
            zAug_projected = projector(zAug)
            contrasLoss = nt_xent_loss(z_projected, zAug_projected, temperature=0.5)
            # contrasLoss = torch.tensor(2.0)
            
            # ------ contrastive loss 2 ------
            # Before loss calculation we have to remove the target label related outputs.
            # nontarget is the list of indices related to source domains (non-target domain)
            
            nontarget = torch.where(batch_labels != activity_num)[0]
            batch_labels_source = batch_labels[nontarget]
            z_projected_source = z_projected[nontarget]
            contrasLoss2 = activity_contrastive_loss(z_projected_source, batch_labels_source, temperature=0.5)
            # contrasLoss2 = torch.tensor(2.0)
            
            # ==========Total loss=============
            loss = recon_weight * (reconLoss+reconAugLoss) + contrastive_weight * (contrasLoss + contrasLoss2) + kld
            # loss = recon_weight * (reconLoss+reconAugLoss) + kld

            # ======== Update Gradients =========
            opt_encoder.zero_grad()
            opt_decoder.zero_grad()
            opt_projector.zero_grad()
            loss.backward()
            opt_projector.step()
            opt_encoder.step()
            opt_decoder.step()
            
            
            trainLoss += loss.item()
            traintotal += batch_samples.size(0)
            
            contrasTotal += contrasLoss.item()
            contrasTotal2 += contrasLoss2.item()
            reconLossRegTotal += reconLoss.item()
            reconLossAugTotal += reconAugLoss.item()
            kldloss += kld.item()
            
            encoder.eval()
            decoder.eval()
        #====================
        # Validation
        #====================
        with torch.no_grad():
            validLoss = 0.0
            validtotal = 0
            vreconLossTotal = 0.0
            vreconLossAugTotal = 0.0
            vcontrasTotal = 0.0
            vcontrasTotal2 = 0.0
            vkldloss = 0.0
            
            for vidx, (aug_t_sample, t_sample, _) in enumerate(T_valid):
                aug_t_sample, t_sample = aug_t_sample.to(DEVICE).float(), t_sample.to(DEVICE).float()
                for i in range(0, nsources):
                    try:
                        sampleAug[i], sample[i], label[i] = next(S_valid_itr_list[i])
                    except StopIteration:
                        S_valid_itr = iter(S_valid[i])
                        sampleAug[i], sample[i], label[i] = next(S_valid_itr)
                    sampleAug[i], sample[i], label[i] = sampleAug[i].to(DEVICE).float(), sample[i].to(DEVICE).float(), label[i].to(DEVICE).long()
                
                batch_sample_list = []
                batch_augsample_list = []
                batch_label_list = []
                
                for i in range(0, nsources):
                    batch_sample_list.append(sample[i])
                    batch_augsample_list.append(sampleAug[i])
                    batch_label_list.append(label[i])
                    
                batch_sample_list.append(t_sample)
                batch_augsample_list.append(aug_t_sample)
                batch_label_list.append(torch.full((t_sample.size(0),), activity_num).to(DEVICE).long())

                batch_samples = torch.cat(batch_sample_list, dim=0)
                batch_augsamples = torch.cat(batch_augsample_list, dim=0)
                batch_labels = torch.cat(batch_label_list, dim=0)

                perm_indices = torch.randperm(batch_samples.size(0))
                # Shuffle tensors based on the generated indices
                batch_samples = batch_samples[perm_indices]
                batch_augsamples = batch_augsamples[perm_indices]
                batch_labels = batch_labels[perm_indices]
                
                vmu, vlogvar = encoder(batch_samples)
                vmuAug, vlogvarAug = encoder(batch_augsamples)
                vz, vmu, vstd = reparameterize(vmu, vlogvar)
                vzAug, vmu, vstdAug = reparameterize(vmuAug, vlogvarAug)
                
                #===========decoder===============
                vreconSample = decoder(vz)
                vreconSampleAug = decoder(vzAug)
                
                #=======reconstruction loss=======
                vreconLoss = criterion(vreconSample, batch_samples)
                vreconAugLoss = criterion(vreconSampleAug, batch_augsamples)
                
                #==========KL divergence==========
                vkld = -0.5 * torch.sum(1 + vlogvar - vmu.pow(2) - vlogvar.exp())
                vkldAug = -0.5 * torch.sum(1 + vlogvarAug - vmuAug.pow(2) - vlogvarAug.exp())
                
                #=========contrastive loss=========
                vz_projected = projector(vz)
                vzAug_projected = projector(vzAug)
                vcontrasLoss = nt_xent_loss(vz_projected, vzAug_projected, temperature=0.5)
                # vcontrasLoss = torch.tensor(5.0)
                
                # ------ contrastive loss 2 ------
                # Before loss calculation we have to remove the target label related outputs.
                # nontarget is the list of indices related to source domains (non-target domain)
                
                nontarget = torch.where(batch_labels != activity_num)[0]
                vbatch_labels_source = batch_labels[nontarget]
                vz_projected_source = vz_projected[nontarget]
                vcontrasLoss2 = activity_contrastive_loss(vz_projected_source, vbatch_labels_source, temperature=0.5)
                # vcontrasLoss2 = torch.tensor(3.0)
                
                #==========Total loss==============
                vLoss = recon_weight * (vreconLoss+vreconAugLoss) + contrastive_weight * (vcontrasLoss + vcontrasLoss2) + vkld #+ vkldAug
                # vLoss = recon_weight * (vreconLoss+vreconAugLoss) + vkld
                
                validtotal += batch_samples.size(0)
                validLoss += vLoss.item()
                vreconLossTotal += vreconLoss.item()
                vreconLossAugTotal += vreconAugLoss.item()
                vcontrasTotal += vcontrasLoss.item()
                vcontrasTotal2 += vcontrasLoss2.item()
                vkldloss += vkld.item()
                
        # Step the scheduler at the end of each epoch
        scheduler_encoder.step()
        scheduler_decoder.step()
        scheduler_projector.step()
        
        avgTrainLoss = trainLoss / traintotal
        avgValidLoss = validLoss / validtotal
        avgTrainContrasLoss = contrasTotal/traintotal
        avgValidContrasLoss = vcontrasTotal/validtotal
        avgTrainContrasLoss2 = contrasTotal2/traintotal
        avgValidContrasLoss2 = vcontrasTotal2/validtotal
        avgTrainReconLoss = reconLossRegTotal/traintotal
        avgValidReconLoss = vreconLossTotal/validtotal
        avgTrainKLD = kldloss/traintotal
        avgValidKLD = vkldloss/validtotal
        
        print(f'Epoch: {epoch+1}/{N_EPOCH}')
        print(f"Average Training Loss: {avgTrainLoss}")
        print(f"Average Validation Loss: {avgValidLoss}")
        print("=======================================")
        print(f"Average Training Contras(z_projected, zAug_projected) loss is: {avgTrainContrasLoss}")
        print(f"Average valid Contras(vz_projected, vzAug_projected) loss is: {avgValidContrasLoss}")
        print("=======================================")
        print(f"Average Training Contras(z_projected, labels) loss is: {avgTrainContrasLoss2}")
        print(f"Average valid Contras(z_projected, labels) loss is: {avgValidContrasLoss2}")
        print("=======================================")
        print(f"Average Training recon reg loss is: {avgTrainReconLoss}")
        print(f"Average valid recon reg loss is: {avgValidReconLoss}")
        print("=======================================")
        print(f"Average Training recon aug loss is: {reconLossAugTotal/traintotal}")
        print(f"Average valid recon aug loss is: {vreconLossAugTotal/validtotal}")
        print("=======================================")
        print(f"Average Training kld loss is: {avgTrainKLD}")
        print(f"Average valid kld loss is: {avgValidKLD}")
        
        log_file.write(f'Epoch:{epoch}\n')
        log_file.write(f'Train_loss:{avgTrainLoss}' + f' Train_Contrastive_loss:{avgTrainContrasLoss}' + f' Train_Recon_Loss:{avgTrainReconLoss}' + f' Train_Contrastive_loss2:{avgTrainContrasLoss2}' + f' Train_KLD:{avgTrainKLD}' + '\n')
        log_file.write(f'Valid_loss:{avgValidLoss}' + f' Valid_Contrastive_loss:{avgValidContrasLoss}' + f' Valid_Recon_Loss:{avgValidReconLoss}' + f' Valid_Contrastive_loss2:{avgValidContrasLoss2}' + f' Valid_KLD:{avgValidKLD}' + '\n')
        torch.save(encoder.state_dict(), encoder_save_path)
    log_file.close()




def trainCLF(save_path, sources, encoder, classifier, clf_epoch, S_train, S_valid, S_test, scheduler_clf, optimizer, DEVICE):
    print("\n------------------------------------------------")
    print("------------ CLASSIFIER TRAINING ---------------")
    print("------------------------------------------------")
    nsources = len(sources)
    # =========================
    # Loss
    # =========================
    criterion_classifier = nn.CrossEntropyLoss()
    # =========================
    # Iterator Setup
    # =========================
    sampleAug = [0] * nsources
    sample = [0] * nsources
    label = [0] * nsources

    S_train_itr_list = []
    S_valid_itr_list = []
    S_test_itr_list = []
            
    for i in range(0, nsources):
        S_train_itr_list.append(iter(S_train[i]))
        S_valid_itr_list.append(iter(S_valid[i]))
        S_test_itr_list.append(iter(S_test[i]))
    #=========================================
    # Training Log
    #=========================================
    log_filename = save_path + "classifier_log.txt"
    log_file = open(log_filename, "w")
    prevValidAcc = None #valid accuracy in previous epoch

    #==========Encoder Weights Freeze============
    encoder.requires_grad_(False)
    #================================
    # Classifier Training
    #================================

    # Training loop
    for epoch in range(clf_epoch):
        classifier.train()
        classifierTrainLoss = 0.0
        task_correct = 0
        task_total = 0
        task_train_acc = 0
        for batch_idx, (sampleAug[0], sample[0], label[0]) in enumerate(S_train[0]):
            sampleAug[0], sample[0], label[0] = sampleAug[0].to(DEVICE).float(), sample[0].to(DEVICE).float(), label[0].to(DEVICE).long()
            for i in range(1, nsources):
                try:
                    sampleAug[i], sample[i], label[i] = next(S_train_itr_list[i-1])
                except StopIteration:
                    S_train_itr = iter(S_train[i])
                    sampleAug[i], sample[i], label[i] = next(S_train_itr)
                    
                sampleAug[i], sample[i], label[i] = sampleAug[i].to(DEVICE).float(), sample[i].to(DEVICE).float(), label[i].to(DEVICE).long()
            
            batch_sample_list = []
            batch_augsample_list = []
            batch_label_list = []
            for i in range(0, nsources):
                batch_sample_list.append(sample[i])
                batch_augsample_list.append(sampleAug[i])
                batch_label_list.append(label[i])
                
            batch_samples = torch.cat(batch_sample_list, dim=0)
            batch_augsamples = torch.cat(batch_augsample_list, dim=0)
            batch_labels = torch.cat(batch_label_list, dim=0)
            
            perm_indices = torch.randperm(batch_samples.size(0))
            # Shuffle tensors based on the generated indices
            batch_samples = batch_samples[perm_indices]
            batch_augsamples = batch_augsamples[perm_indices]
            batch_labels = batch_labels[perm_indices]
            
            #======Get mean and logvar=========
            mu, logvar = encoder(batch_samples)
            muAug, logvarAug = encoder(batch_augsamples)
            
            #======Reparameterization trick=======
            z, mu, std = reparameterize(mu, logvar)
            zAug, muAug, stdAug = reparameterize(muAug, logvarAug)
            
            #=========Classifier training========
            outputs = classifier(z.detach())
            outputsAug = classifier(zAug.detach())
            
            loss_classifier = criterion_classifier(outputs, batch_labels)
            loss_classifier_aug = criterion_classifier(outputsAug, batch_labels)
            classifierTrainLoss += loss_classifier.item()
            classifierTrainLoss += loss_classifier_aug.item()
            
            optimizer.zero_grad()
            loss_classifier.backward()
            loss_classifier_aug.backward()
            optimizer.step()
            
            _, task_pred = torch.max(outputs.data, 1)
            task_correct += (task_pred == batch_labels).sum()
            task_total += outputs.size(0)
            
        classifier.eval()
        with torch.no_grad():
            validLoss = 0.0
            validtotal = 0
            valid_correct = 0
            for vidx, (sampleAug[0], sample[0], label[0]) in enumerate(S_valid[0]):
                sampleAug[0], sample[0], label[0] = sampleAug[0].to(DEVICE).float(), sample[0].to(DEVICE).float(), label[0].to(DEVICE).long()
                for i in range(1, nsources):
                    try:
                        sampleAug[i], sample[i], label[i] = next(S_valid_itr_list[i-1])
                    except StopIteration:
                        S_valid_itr = iter(S_train[i])
                        sampleAug[i], sample[i], label[i] = next(S_valid_itr)

                    sampleAug[i], sample[i], label[i] = sampleAug[i].to(DEVICE).float(), sample[i].to(DEVICE).float(), label[i].to(DEVICE).long()

                batch_sample_list = []
                batch_augsample_list = []
                batch_label_list = []
                for i in range(0, nsources):
                    batch_sample_list.append(sample[i])
                    batch_augsample_list.append(sampleAug[i])
                    batch_label_list.append(label[i])

                batch_samples = torch.cat(batch_sample_list, dim=0)
                batch_augsamples = torch.cat(batch_augsample_list, dim=0)
                batch_labels = torch.cat(batch_label_list, dim=0)

                perm_indices = torch.randperm(batch_samples.size(0))
                # Shuffle tensors based on the generated indices
                batch_samples = batch_samples[perm_indices]
                batch_augsamples = batch_augsamples[perm_indices]
                batch_labels = batch_labels[perm_indices]
                
                
                vmu, vlogvar = encoder(batch_samples)
                vmuAug, vlogvarAug = encoder(batch_augsamples)
                
                vz, vmu, vstd = reparameterize(vmu, vlogvar)
                vzAug, vmuAug, vstdAug = reparameterize(vmuAug, vlogvarAug)
                
                out = classifier(vz)
                outAug = classifier(vzAug)
                
                vLoss = criterion_classifier(out, batch_labels)
                vLoss += criterion_classifier(outAug, batch_labels)
                
                validLoss += vLoss
                validtotal += batch_samples.size(0)
                
                _, valid_pred = torch.max(out.data, 1)
                valid_correct += (valid_pred == batch_labels).sum()
        
        # Step the scheduler at the end of each epoch
        scheduler_clf.step()

        #=======Print or log the average loss for the epoch========
        avgTrainLossCLF = classifierTrainLoss / task_total
        avgValidLossCLF = validLoss / validtotal
        validAcc = (float(valid_correct)*100 / validtotal)
        trainAcc = (float(task_correct) * 100/ task_total)
        
        print("=======================================")
        print(f'Epoch [{epoch + 1}/{clf_epoch}], Training Loss: {avgTrainLossCLF}')
        print(f"Epoch [{epoch + 1}/{clf_epoch}], Validation Loss: {avgValidLossCLF}")
        print(f'Epoch [{epoch + 1}/{clf_epoch}], Training Accuracy: {trainAcc}')
        print(f'Epoch [{epoch + 1}/{clf_epoch}], Validation Accuracy: {validAcc}')
        
        log_file.write(f'Epoch:{epoch}\n')
        log_file.write(f'classifier_train_loss:{avgTrainLossCLF}' + f' classifier_valid_loss:{avgValidLossCLF}' + "\n")
        log_file.write(f'classifier_train_Accuracy:{trainAcc}' + f' classifier_valid_Accuracy:{validAcc}' + "\n")
        classifier_save_path = save_path + "classifier.pth"
        
        if prevValidAcc == None:
            torch.save(classifier.state_dict(), classifier_save_path)
            prevValidAcc = validAcc
        elif validAcc >= prevValidAcc:
            torch.save(classifier.state_dict(), classifier_save_path)
            prevValidAcc = validAcc
        
    log_file.close()