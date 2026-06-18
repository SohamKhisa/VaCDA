import matplotlib.pyplot as plt

def plot_losses(log_file_path, losses_to_plot, plot_save_dir):
    # Initialize dictionaries to store losses
    train_losses = {
        'Train_loss': [],
        'Train_Contrastive_loss': [],
        'Train_Recon_Loss': [],
        'Train_Contrastive_loss2': [],
        'Train_KLD': []
    }
    valid_losses = {
        'Valid_loss': [],
        'Valid_Contrastive_loss': [],
        'Valid_Recon_Loss': [],
        'Valid_Contrastive_loss2': [],
        'Valid_KLD': []
    }
    epochs = []

    # Read the log file and extract losses
    with open(log_file_path, 'r') as f:
        for line in f:
            if line.startswith('Epoch'):
                epoch = int(line.split(':')[1])
                epochs.append(epoch)
            elif line.startswith('Train_loss'):
                parts = line.split()
                for part in parts:
                    key, value = part.split(':')
                    train_losses[key].append(float(value))
            elif line.startswith('Valid_loss'):
                parts = line.split()
                for part in parts:
                    key, value = part.split(':')
                    valid_losses[key].append(float(value))

    # Plot the requested losses
    plt.figure(figsize=(10, 6))
    plot_name_parts = []

    for loss_name in losses_to_plot:
        if loss_name in train_losses:
            plt.plot(epochs, train_losses[loss_name], label=loss_name)
            plot_name_parts.append(loss_name)
        if loss_name in valid_losses:
            plt.plot(epochs, valid_losses[loss_name], label=loss_name)
            plot_name_parts.append(loss_name)

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    plt.grid(True)
    
    # Generate unique plot name
    plot_name = '-'.join(plot_name_parts) + '.png'
    plot_save_dir = plot_save_dir + plot_name
    plt.savefig(plot_save_dir)
    
    # Show the plot
    plt.close()



def plot_classifier_metrics(log_file_path, save_dir):
    # Initialize lists to hold values
    train_loss = []
    valid_loss = []
    train_acc = []
    valid_acc = []
    
    # Read the log file
    with open(log_file_path, 'r') as log_file:
        lines = log_file.readlines()
    
    for line in lines:
        if 'classifier_train_loss' in line:
            train_loss.append(float(line.split('classifier_train_loss:')[1].split()[0]))
            valid_loss.append(float(line.split('classifier_valid_loss:')[1].strip()))
        elif 'classifier_train_Accuracy' in line:
            train_acc.append(float(line.split('classifier_train_Accuracy:')[1].split()[0]))
            valid_acc.append(float(line.split('classifier_valid_Accuracy:')[1].strip()))
    
    # Plot training and validation loss
    plt.figure()
    plt.plot(train_loss, label='Train Loss')
    plt.plot(valid_loss, label='Valid Loss')
    plt.title('Classifier Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plot_name_loss = "Classifier_TrainLoss_ValidLoss.png"
    clf_loss_save = save_dir + plot_name_loss
    plt.savefig(clf_loss_save)
    plt.close()
    
    # Plot training and validation accuracy
    plt.figure()
    plt.plot(train_acc, label='Train Accuracy')
    plt.plot(valid_acc, label='Valid Accuracy')
    plt.title('Classifier Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plot_name_acc = "Classifier_TrainAcc_ValidAcc.png"
    clf_acc_save = save_dir + plot_name_acc
    plt.savefig(clf_acc_save)
    plt.close()