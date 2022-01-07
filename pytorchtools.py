import os

import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, save_Path = ''):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.train_loss_min = np.Inf
        self.save_Path = save_Path

    def __call__(self, train_loss, fu, sr, save_Path):

        score = -train_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(train_loss, fu, sr, save_Path)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(train_loss, fu, sr, save_Path)
            self.counter = 0

    def save_checkpoint(self, train_loss, fu, sr, save_Path):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'train loss decreased ({self.train_loss_min:.6f} --> {train_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), '{}_checkpoint.pt'.format(self.save_name))
        save_ckpt({
            'Sec_Fusionnet_state_dict': fu.state_dict(),
            'srnet_state_dict': sr.state_dict(),
            # 'netD_state_dict': netD.state_dict(),
            # 'loss_tex': loss_tex,
            # 'loss_sum': loss_sum,
        }, save_path=save_Path,
            filename='fu_srnet' + '.pth.tar')
        self.train_loss_min = train_loss

def save_ckpt(state, save_path='./log/x4', filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(save_path,filename))

class print_all_EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, save_name = ''):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.save_name = save_name
        self.epoch = 0

    def __call__(self, val_loss, model):

        score = -val_loss
        self.epoch += 1
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            self.save_checkpoint(val_loss, model, False)
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        # self.save_checkpoint(val_loss, model)

    def save_checkpoint(self, val_loss, model, state = True):
        '''Saves model when validation loss decrease.'''
        if self.verbose and state:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), '{}_checkpoint_{}.pt'.format(self.save_name, self.epoch))
        if state:
            self.val_loss_min = val_loss
