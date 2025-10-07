# SETUP, IMPORTS, ETC
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import os
from torch import nn
import torch.nn.functional as F

#from torch.utils.data import Dataset
#import torch.nn.functional as F
#import torchvision.transforms as tvtf

class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, X, Y, weights=None):
        'Initialization'
        self.X = X
        self.Y = Y
        if weights is None:
            weights = np.ones(Y.shape)
        self.weights = weights

    def __len__(self):
        'Denotes the total number of samples'
        return self.X.shape[0]

    def __getitem__(self, index):
        'Generates one sample of data'

        # Load data and get label
        Xind = self.X[index,:]
        Yind = self.Y[index]
        weightsind = self.weights[index]

        return Xind, Yind, weightsind


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class DA_Net(nn.Module):
    def __init__(self, dropout=0, train_noise=0, n_feat=500, n_neur=10, device='cpu', model_type='fc', out_scale=1):
        super(DA_Net, self).__init__()
        self.Flatten  = Flatten()
        self.device = device
        self.model_type = model_type
        
        self.fc0 = nn.Linear(n_feat,n_neur)
        self.fc1 = nn.Linear(n_neur,n_neur)
        self.fc2 = nn.Linear(n_neur,n_neur)
        self.fc3 = nn.Linear(n_neur,n_neur)
        self.fc4 = nn.Linear(n_neur,int(n_neur/2))
        self.fc5 = nn.Linear(int(n_neur/2),int(n_neur/4))
        self.fc6 = nn.Linear(int(n_neur/4)+2,int(n_neur/4))
        self.fc_out = nn.Linear(int(n_neur/4),1)
        self.dropout = nn.Dropout(dropout)
        self.mbn = nn.BatchNorm1d(n_neur, affine=False)
        self.flatten = nn.Flatten()
        self.train_noise = train_noise
        self.out_scale = out_scale

    def forward(self, x):
        #x = self.forward_fc(x)
        if self.model_type=='fc':
            x = self.forward_fc(x)
        if self.model_type=='split':
            x = self.forward_split(x)
        if self.model_type=='sine':
            x = self.forward_split(x)
        return x

    def forward_fc(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc0(x))
        #x = self.dropout(x)
        x = F.relu(self.fc2(x))
        #x = self.dropout(x)
        x = F.relu(self.fc3(x))
        #x = self.dropout(x)
        x = F.relu(self.fc4(x))
        #x = self.dropout(x)
        x = F.relu(self.fc5(x))
        x = self.out_scale * torch.sigmoid(self.fc_out(x))
        #x = self.fc_out(x)
        return x

    def forward_split(self, x):
        x1 = x[:,:-2]
        x2 = x[:,-2:]
        x = self.flatten(x)
        x = F.relu(self.fc0(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        x = self.dropout(x)
        x = F.relu(self.fc5(x))
        x = torch.cat((x,x2), dim=1)
        x = F.relu(self.fc6(x))
        x = self.out_scale * torch.sigmoid(self.fc_out(x))
        #x = self.fc_out(x)
        return x


    def forward_sine(self, x):
        x1 = x[:,:-2]
        x2 = x[:,-2:]
        x = self.flatten(x)
        x = torch.sin(self.fc0(x))
        #x = self.dropout(x)
        x = torch.sin(self.fc2(x))
        #x = self.dropout(x)
        x = torch.sin(self.fc3(x))
        #x = self.dropout(x)
        x = torch.sin(self.fc4(x))
        #x = self.dropout(x)
        x = torch.sin(self.fc5(x))
        x = torch.cat((x,x2), dim=1)
        x = torch.sin(self.fc6(x))
        x = self.out_scale * torch.sigmoid(self.fc_out(x))
        #x = self.fc_out(x)
        return x

def get_norm(X, eps = 1e-5):

    nvar = X.shape[1]
    X_mu = np.zeros(nvar)
    X_std = np.zeros(nvar)
    for j in range(nvar):
        Xj = X[:,j]
        X_mu[j] = np.mean(Xj)
        X_std[j] = np.std(Xj) + eps
    return X_mu, X_std

def normalize(X, X_mu, X_std):

    nvar = X.shape[1]
    for j in range(nvar):
        Xj = X[:,j]
        Xj_norm = (Xj - X_mu[j])/X_std[j]
        X[:,j] = Xj_norm
    return X

def myloss(out, y, weights):
    loss = torch.mean(weights * (out - y)**2 / torch.mean(weights))
    #loss = torch.mean(torch.abs(out - y))
    return(loss)

def infer(net, x, nsamp = 10):
    
    nex = x.shape[0]
    out = np.zeros([nsamp, nex], dtype=np.float16)
    for j in range(nsamp):
        outj = torch.squeeze(net(x.float())).detach().to('cpu').numpy()
        out[j, :] = outj
        
    return(out)
    
def predict_turns(X, danet, nsamp=10, max_pts=50000, device='cpu', verbose=1):

    t0 = time.time()
    npts = X.shape[0]
    turns = np.zeros([nsamp,npts], dtype = np.float16)
    nchunk = int(np.ceil(npts/max_pts))
    if verbose > 1:
        print('Begin inference on %d chunks' % nchunk)
    for j in range(nchunk):
        j0 = j * max_pts
        j1 = np.min([(j+1) * max_pts, npts])
        turns_j = infer(danet, torch.from_numpy(X[j0:j1,:]).to(device), nsamp=nsamp)
        turns[:,j0:j1] = turns_j
        if ((j+1)%10 == 0 or j == 0) and verbose>0:
            print('finished chunk %d in %d sec' % (j+1,time.time()-t0))
        
    return(turns)


def predict_turns_compact(X, danet, nsamp=10, max_pts=50000, device='cpu', mytype='single', verbose=1):
    # Trying to reduce memory by only saving statistics, but may be worse for meanBAX

    t0 = time.time()
    npts = X.shape[0]
#    turns = np.zeros([nsamp,npts])
    turns_mean = np.zeros([1,npts], dtype=mytype)
    turns_std = np.zeros([1,npts], dtype=mytype)
    nchunk = int(npts/max_pts)
    if verbose > 1:
        print('Begin inference on %d chunks' % nchunk)
    for j in range(nchunk):
        j0 = j * max_pts
        j1 = np.min([(j+1) * max_pts, npts])
#        turns[:,j0:j1] = infer(danet, torch.from_numpy(X[j0:j1,:]).to(device), nsamp = nsamp).detach().numpy()
        turns_temp = infer(danet, torch.from_numpy(X[j0:j1,:]).to(device), nsamp = nsamp).detach().numpy()
        turns_mean[0,j0:j1] = np.mean(turns_temp.astype(mytype), axis=0)
        turns_std[0,j0:j1] = np.std(turns_temp.astype(mytype), axis=0)
        if verbose > 0:
            print('finished chunk %d in %d sec' % (j+1,time.time()-t0))
        
    turns = np.concatenate((turns_mean,turns_std), axis=0)
    return(turns)

def train_NN(net, trainloader, testloader, lr=1e-4, epochs=10, savefile=None, device='cpu',
             verbose=1, log=None):

    # Define optimizer.  Haven't played around with optimizers
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    net = net.float()

    best_score=0     # remember best loss for saving models.  Should update this to check when reloading
    print('Training...', flush=True)
    
    t0=time.time()
    
    running_loss = 0.0
    for epoch in range(epochs):

        for i, data in enumerate(trainloader):
            x, y, weights = data
            x = x.to(device)
            y = y.to(device)
            weights = weights.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            #torch.autograd.set_detect_anomaly(True)
            out = torch.squeeze(net(x.float()))

            loss = myloss(out, y, weights)

            # Update model
            loss=loss.to(device)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            fudge=1e0
            if (i+1)%10==0 or ((epoch+1)%1 == 0 and i == 0) or (epoch == 0 and i == 0):
                x_test, y_test, weights_test = next(iter(testloader))
                x_test = x_test.to(device)
                y_test = y_test.to(device)
                weights_test = weights_test.to(device)
                out_test=torch.squeeze(net(x_test.float()))
                test_loss = myloss(out_test, y_test, weights_test)
                if verbose > 0:
                    print('%d sec, batch %d-%d, train loss: %.4f, test loss: %.4f' % ((time.time()-t0),epoch+1,i+1,loss,test_loss),flush=True)
                    if log is not None:
                        log.append([epoch + 1, i + 1, loss, test_loss])


            # Save model if starting out or if new best score observed
            if epoch==0 or test_loss < best_score:
                torch.save(net.state_dict(), savefile)
                best_score = test_loss


    return(net)


def train_NN_re(net, trainloader, testloader, lr=1e-4, epochs=10, 
                savefile=None, final_savefile=None, device='cpu',
                early_stop_patience=None,  # turn off early stopping by default
                verbose=1, log_period=10, log=None,
                eval_mode_on_test=True, use_batch_loss=False):

    # Define optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    
    # Move model to device and cast to float
    net = net.float().to(device)
    
    best_score = float('inf')
    epochs_without_improvement = 0
    print('Training...', flush=True)
    t0 = time.time()

    for epoch in range(epochs):
        # Save the best score at the start of the epoch
        prev_best = best_score

        if eval_mode_on_test:
            net.train()  # Set model to training mode
        running_loss = 0.0    # Accumulate weighted training loss (loss * number of samples)
        running_samples = 0   # Total number of training samples processed in current logging period

        for i, data in enumerate(trainloader):
            x, y, weights = data
            x, y, weights = x.to(device), y.to(device), weights.to(device)
            batch_size = x.size(0)

            optimizer.zero_grad()
            # Forward pass: cast inputs to float and squeeze the output if needed
            out = torch.squeeze(net(x.float()))
            loss = myloss(out, y, weights)
            loss.backward()
            optimizer.step()

            # Accumulate weighted loss for this batch
            running_loss += loss.item() * batch_size
            running_samples += batch_size

            # Log every 10 batches (including the first batch when i == 0)
            if i % log_period == 0:
                # Weighted average training loss
                avg_train_loss = running_loss / running_samples if running_samples > 0 else 0.0
                
                # Use loss on a single batch instead of averaging loss on multiple batches
                if use_batch_loss:
                    loss_single = loss.item()

                # Evaluate on the full test set
                if eval_mode_on_test:
                    net.eval()  # Switch to evaluation mode
                    
                total_test_loss = 0.0
                total_test_samples = 0
                first_batch_flag = True
                with torch.no_grad():
                    for x_test, y_test, weights_test in testloader:
                        x_test, y_test, weights_test = (
                            x_test.to(device),
                            y_test.to(device),
                            weights_test.to(device)
                        )
                        batch_size_test = x_test.size(0)
                        out_test = torch.squeeze(net(x_test.float()))
                        test_loss_batch = myloss(out_test, y_test, weights_test)
                        
                        if first_batch_flag and use_batch_loss:
                            test_loss_single = test_loss_batch.item()
                            first_batch_flag = False
                            
                        total_test_loss += test_loss_batch.item() * batch_size_test
                        total_test_samples += batch_size_test
                avg_test_loss = total_test_loss / total_test_samples if total_test_samples > 0 else 0.0

                # Print and log the current losses
                if verbose > 0:
                    elapsed = time.time() - t0
                    if use_batch_loss:
                        print(f'{elapsed:.2f} sec, epoch {epoch+1}, batch {i+1}, '
                              f'train loss: {loss_single:.4f}, test loss: {test_loss_single:.4f}',
                              flush=True)
                    else:
                        print(f'{elapsed:.2f} sec, epoch {epoch+1}, batch {i+1}, '
                              f'train loss: {avg_train_loss:.4f}, test loss: {avg_test_loss:.4f}',
                              flush=True)
                if log is not None:
                    if use_batch_loss:
                        log.append([epoch + 1, i + 1, loss_single, test_loss_single])
                    else:
                        log.append([epoch + 1, i + 1, avg_train_loss, avg_test_loss])

                # Save model immediately if test loss is improved
                if savefile is not None and avg_test_loss < best_score:
                    torch.save(net.state_dict(), savefile)
                    best_score = avg_test_loss

                # Reset accumulators after logging
                running_loss = 0.0
                running_samples = 0

                if eval_mode_on_test:
                    net.train()  # Switch back to training mode

        # End-of-epoch evaluation on the test set
        if eval_mode_on_test:
            net.eval()
        total_test_loss = 0.0
        total_test_samples = 0
        with torch.no_grad():
            for x_test, y_test, weights_test in testloader:
                x_test, y_test, weights_test = (
                    x_test.to(device),
                    y_test.to(device),
                    weights_test.to(device)
                )
                batch_size_test = x_test.size(0)
                out_test = torch.squeeze(net(x_test.float()))
                test_loss_batch = myloss(out_test, y_test, weights_test)
                total_test_loss += test_loss_batch.item() * batch_size_test
                total_test_samples += batch_size_test
        avg_test_loss = total_test_loss / total_test_samples if total_test_samples > 0 else 0.0
        
        if verbose > 0:
            print(f'End of epoch {epoch+1}: avg test loss: {avg_test_loss:.4f}', flush=True)
        
        # Check improvement at epoch end: update best_score if improved
        if avg_test_loss < best_score:
            best_score = avg_test_loss
            if savefile is not None:
                torch.save(net.state_dict(), savefile)
        
        # Determine if there was improvement this epoch compared to the start
        if best_score < prev_best:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Early stopping check
        if early_stop_patience is not None and early_stop_patience > 0:
            if epochs_without_improvement >= early_stop_patience:
                if verbose > 0:
                    print(f'Early stopping triggered at epoch {epoch+1}', flush=True)
                break

    # After training, save final model if final_savefile is provided
    if final_savefile is not None:
        torch.save(net.state_dict(), final_savefile)

    return net
