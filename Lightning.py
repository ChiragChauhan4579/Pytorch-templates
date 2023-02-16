import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning import Trainer


input_size = 
hidden_size = 
num_classes = 
num_epochs = 
batch_size = 
learning_rate = 

class LitNeuralNet(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LitNeuralNet, self).__init__()
        
        # all self.layers to be included

    def forward(self, x):
      
        #network architecture
        out = self.layers(x)
        return out

    def training_step(self, batch, batch_idx):
      
      # training loop and calculate loss

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader()
        return train_loader

    def val_dataloader(self):
        test_loader = torch.utils.data.DataLoader()
        return test_loader
    
    def validation_step(self, batch, batch_idx):
        
        # validation loss
    
    def validation_epoch_end(self, outputs):
        
        # epoch validation avg loss
    
    
    def configure_optimizers(self):
        return # optimizer
      
model = LitNeuralNet(input_size, hidden_size, num_classes)
trainer = Trainer(max_epochs=num_epochs)
trainer.fit(model)
