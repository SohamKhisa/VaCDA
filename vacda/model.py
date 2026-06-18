import torch.nn as nn
import torch.nn.functional as F

# Takes from input from the latent representation
#--------------------------------------------------#
#------------------ CLASSIFIER --------------------#
#--------------------------------------------------#
class Classifier(nn.Module):
    def __init__(self, gt_size, infeature=64):
        super(Classifier, self).__init__()
        self.gt_size = gt_size
        
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=infeature, out_features=1024),
            nn.LeakyReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=1024, out_features=256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=256, out_features=self.gt_size)
        )
                    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

#---------------------------------------------------#
#-------------------- ENCODER ----------------------#
#---------------------------------------------------#
class Encoder(nn.Module):
    def __init__(self, latent_dim, init_channel, last_channel):
        super(Encoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.init_channel = init_channel
        self.last_channel = last_channel
        
        self.conv1 = nn.Conv2d(in_channels=init_channel, out_channels=256, kernel_size=(1,3)) # torch.Size([batchsize, 256, 1, 98])
        self.bn1 = nn.BatchNorm2d(num_features=256)
        
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1,3)) # torch.Size([batchsize, 128, 1, 96])
        self.bn2 = nn.BatchNorm2d(num_features=128)
        
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=last_channel, kernel_size=(1,3)) # torch.Size([batchsize, 64, 1, 94])
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        
        # after flattening [batchsize, latent_dim*94]
        self.fc_mean = nn.Linear(in_features=self.last_channel*94, out_features=self.latent_dim)
        self.fc_logvar = nn.Linear(in_features=self.last_channel*94, out_features=self.latent_dim)
 

    def forward(self, x):
        # x --> torch.Size([batchsize, 3, 1, win_size])
        # print(x.shape)
        x = F.relu(self.bn1(self.conv1(x)))
        # print(x.shape)
        x = F.relu(self.bn2(self.conv2(x)))
        # print(x.shape)
        x = self.lrelu(self.conv3(x))
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        mu = self.fc_mean(x)
        log_var = self.fc_logvar(x)
        return mu, log_var
        
# Latent dimension size is 64


#---------------------------------------------------#
#-------------------- DECODER ----------------------#
#---------------------------------------------------#
class Decoder(nn.Module):
    def __init__(self, latent_dim, init_channel, last_channel):
        super(Decoder, self).__init__()
        
        # Decoder with transposed convolutions
        self.latent_dim = latent_dim
        self.init_channel = init_channel
        self.last_channel = last_channel
        
        self.dense = nn.Linear(in_features=latent_dim, out_features=self.init_channel*94)
        self.deconv1 = nn.ConvTranspose2d(in_channels=self.init_channel, out_channels=128, kernel_size=(1, 3))  # Upsizes feature maps
        self.bn1 = nn.BatchNorm2d(num_features=128)
        self.relu = nn.ReLU(inplace=True)
        
        self.deconv2 = nn.ConvTranspose2d(in_channels=128, out_channels=256, kernel_size=(1, 3))  # Upsizes feature maps
        self.bn2 = nn.BatchNorm2d(num_features=256)
        
        self.deconv3 = nn.ConvTranspose2d(in_channels=256, out_channels=self.last_channel, kernel_size=(1, 3))  # Recover original size
        self.out = nn.Sigmoid()  # Output between 0 and 1 for image reconstruction

    def forward(self, x):
        # x --> torch.Size([batch, latent_space_dim])
        # print(x.shape)
        x = self.dense(x) # [batch, 64*94]
        x = x.reshape(-1, self.init_channel, 1, 94) # torch.size([batch, 64, 1, 94])
        # print(x.shape)
        x = self.relu(self.bn1(self.deconv1(x)))  # torch.Size([batch, 128, 1, 96])
        # print(x.shape)
        x = self.relu(self.bn2(self.deconv2(x)))  # torch.Size([batch, 256, 1, 98])
        # print(x.shape)
        x = self.out(self.deconv3(x))  # torch.Size([batch, 3, 1, 100])
        # print(x.shape)
        return x


#-----------------------------------------------------------#
#-------------------- PROJECTION-HEAD ----------------------#
#-----------------------------------------------------------#
class ProjectionHead(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, head_type='nonlinear', **kwargs):
        super(ProjectionHead, self).__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.head_type = head_type

        if self.head_type == 'linear':
            self.layers = LinearLayer(self.in_features,self.out_features,False, True)
        elif self.head_type == 'nonlinear':
            self.layers = nn.Sequential(
                LinearLayer(self.in_features, self.hidden_features, True, True),
                nn.ReLU(),
                LinearLayer(self.hidden_features, self.hidden_features, True, True),
                nn.ReLU(),
                LinearLayer(self.hidden_features, self.out_features, False, True)
            )
            
    def forward(self,x):
        x = self.layers(x)
        return x


class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features, use_bias = True, use_bn = False, **kwargs):
        super(LinearLayer, self).__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.use_bn = use_bn
        self.linear = nn.Linear(self.in_features, self.out_features, bias = self.use_bias and not self.use_bn)
        if self.use_bn:
             self.bn = nn.BatchNorm1d(self.out_features)

    def forward(self,x):
        x = self.linear(x)
        if self.use_bn:
            x = self.bn(x)
        return x