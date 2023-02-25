#!/usr/bin/env python
# -*- coding: utf-8 -*-


from torch import cuda
from torch.optim import Adam
from torch.nn import MSELoss
from torch.utils.data import Subset, DataLoader
import numpy as np
import torch
from data_handling import MyDataset
from model.Net import Net
import time
def train():
    # Check if CUDA is available, else use CPU
    device = 'cuda' if cuda.is_available() else 'cpu'
    print(f'Process on {device}', end='\n\n')

    net = Net()

    net.to(device=device, dtype=torch.float)
    
    optimizer = Adam(params=net.parameters(), lr=1e-3)

    loss_function = MSELoss()

    batch_size = 2
    train_dataset = MyDataset("features/train")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Variables for the early stopping
    epochs = 20

    best_model = None
    print("starting training")
    # Start training.
    tr_loss = []
    for epoch in range(epochs):
        # Lists to hold the corresponding losses of each epoch.
        epoch_loss_training = []

        # Indicate that we are in training mode
        net.train()
        start_time = time.time()
        # For each batch of our dataset.
        for i, batch in enumerate(train_loader):
            #print(f"going trough samples {i*batch_size} - {batch_size+i*batch_size} currently taken time {time.time()-start_time}")
            # Zero the gradient of the optimizer.
            optimizer.zero_grad()

            # Get the batches.
            x, y = batch
            # Give them to the appropriate device.
            x = x.to(device)
            y = y.to(device)

            # Get the predictions of our model.
            y_hat = net(x)
            # Calculate the loss of our model.
            loss = loss_function(input=y_hat, target=y.type_as(y_hat))

            # Do the backward pass
            loss.backward()

            # Do an update of the weights (i.e. a step of the optimizer)
            optimizer.step()

            # Loss the loss of the batch
            epoch_loss_training.append(loss.item())
            if (i+1)*batch_size >= 15000: # only 15000 samples
                break
        tr_loss.append(np.array(epoch_loss_training).mean())

        # Indicate that we are in evaluation mode
        net.eval()

        epoch_loss_training = np.array(epoch_loss_training).mean()

        print(f'Epoch: {epoch:03d} | '
              f'Mean training loss: {epoch_loss_training:7.4f}')
        
        torch.save(net.state_dict(), "state.pt")
        

if __name__ == '__main__':
    train()

# EOF