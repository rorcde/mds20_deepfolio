import torch
import time as t
import numpy as np

from model import PADDING_CONST
from utils import check_stopping_criterion

def run_epoch(model, dataloader, device, optimizer=None):
    """
    Runs one epoch of training.

    Input:
        model - THP model,
        dataloader - data loader,
        device - current device,
        optimizer - optimizer to use (if None, then validation run is performed)
    Output:
        ll_loss_epoch - log-likelihood loss value,
        tp_loss_epoch - mean squared error of time prediction,
        ec_loss_epoch - cross entropy loss value,
        accuracy - accuracy value
    """
    
    ll_loss_epoch, tp_loss_epoch, ec_loss_epoch = 0., 0., 0.
    event_num_total, event_pred_correct, event_pred_total = 0, 0, 0

    with torch.set_grad_enabled(optimizer is not None):
        for time, events in dataloader:
            time = time.to(device)
            events = events.to(device)

            h, cond_lam, time_pred, event_logit = model(time, events)

            # Log-likelihood (loss for the whole sequence)
            loss_ll = model.log_likelihood(h, cond_lam, time, events, -0.1, 'mc')

            # Time prediction loss
            loss_tp = model.time_error(time_pred, time)

            # Event type classficiation loss
            loss_ec = model.event_error(event_logit, events)

            # Combined loss function
            scale = 0.01
            loss = -loss_ll + loss_ec + loss_tp * scale

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Logging
            ll_loss_epoch += loss_ll.item()
            tp_loss_epoch += loss_tp.item()
            ec_loss_epoch += loss_ec.item()

            batch_num_events = events.ne(PADDING_CONST).sum().item()
            event_num_total += batch_num_events
            event_pred_correct += (event_logit[:, :-1, :].softmax(dim=2).argmax(dim=2) == events[:, 1:] - 1).sum().item()
            event_pred_total += batch_num_events - events.shape[0]
    
    ll_loss_epoch /= event_num_total
    tp_loss_epoch /= event_pred_total
    ec_loss_epoch /= event_pred_total
    accuracy = event_pred_correct / event_pred_total

    return ll_loss_epoch, tp_loss_epoch, ec_loss_epoch, accuracy

def train(model, n_epochs, optimizer, train_loader, val_loader, scheduler=None, device=None, verbose=True, freq=None,\
          early_stopping=False, stopping_criterion='min_loss', patience=20, checkpoint=True, cp_name=None):
    """
    Training function for the Transformer Hawkes Process model.

    Input:
        model - THP neural network,
        n_epochs (int) - number of epochs to train,
        optimizer - optimizer to use,
        train_loader - train data loader,
        val_loader - validation data loader,
        scheduler - learning rate scheduler,
        device - current device,
        verbose (bool) - output messages or not,
        freq (int) - frequency of output messages (does nothing if verbose is set to False),
        early_stopping (bool) - apply early stopping or not,
        stopping_criterion (string) - which stopping criterion to apply for early stopping and scheduler ('min_loss' or 'max_accuracy'),
        patience (int) - num of epochs where criterion does not improve before training is stopped,
        checkpoint (bool) - apply checkpointing or not (if early stopping is active, then saving is performed only on best iterations),
        cp_name (string) - name for checkpoint
    Output:
        train_history (list) - list of loss and accuracy values on the training set,
        val_history (list) - list of loss and accuracy values on the validation set
    """
    
    if verbose and freq is None:
        freq = max(n_epochs // 10, 1)
    
    if device is None:
        device = GLOBAL_DEVICE
    
    if cp_name is None:
        cp_name = 'model'
    
    assert stopping_criterion != 'min_loss' or stopping_criterion != 'max_accuracy', "Unknown stopping criterion, choose one of 'min_loss' or 'max_accuracy'"

    train_loss_ll_history, train_loss_tp_history, train_loss_ec_history, train_accuracy_history = [], [], [], []
    val_loss_ll_history, val_loss_tp_history, val_loss_ec_history, val_accuracy_history = [], [], [], []

    best_criterion_value = float('inf') if stopping_criterion == 'min_loss' else float('-inf')
    bad_epochs = 0

    time_start = t.time()
    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss_ll, train_loss_tp, train_loss_ec, train_accuracy = run_epoch(model, train_loader, device, optimizer)
        train_loss_ll_history.append( train_loss_ll )
        train_loss_tp_history.append( train_loss_tp )
        train_loss_ec_history.append( train_loss_ec )
        train_accuracy_history.append( train_accuracy )

        model.eval()
        val_loss_ll, val_loss_tp, val_loss_ec, val_accuracy = run_epoch(model, val_loader, device)
        val_loss_ll_history.append( val_loss_ll )
        val_loss_tp_history.append( val_loss_tp )
        val_loss_ec_history.append( val_loss_ec )
        val_accuracy_history.append( val_accuracy )

        criterion_value = -val_loss_ll + val_loss_tp + val_loss_ec if stopping_criterion == 'min_loss' else val_accuracy
        bad_epochs += 1

        if scheduler is not None:
            if type(scheduler).__name__ != 'ReduceLROnPlateau':
                scheduler.step()
            else:
                scheduler.step( criterion_value )
        
        time_epoch_end = t.time() - time_start

        if verbose and epoch % freq == 0:
            print("[ Epoch {} ]".format(epoch))
            print("(Training)     log-likelihood: {}, RMSE: {}, CE: {}, accuracy: {}".format( train_loss_ll, np.sqrt(train_loss_tp), train_loss_ec, train_accuracy ))
            print("(Validation)   log-likelihood: {}, RMSE: {}, CE: {}, accuracy: {}".format( val_loss_ll, np.sqrt(val_loss_tp), val_loss_ec, val_accuracy ))
            print("Time elapsed: {:.2f} s".format(time_epoch_end))
        
        if early_stopping and check_stopping_criterion(criterion_value, best_criterion_value, stopping_criterion):
            bad_epochs = 0
            best_criterion_value = criterion_value

            if checkpoint:
                torch.save(model.state_dict(), '{}.pth'.format(cp_name))
        
        if bad_epochs > patience:
            print("Patience limit for early stopping has been reached, terminating training.")
            break
        
        if not early_stopping and checkpoint:
            torch.save(model.state_dict(), '{}.pth'.format(cp_name))

    train_history = {
        'log-likelihood' : train_loss_ll_history,
        'time mse' : train_loss_tp_history,
        'cross entropy' : train_loss_ec_history,
        'accuracy' : train_accuracy_history
    }

    val_history = {
        'log-likelihood' : val_loss_ll_history,
        'time mse' : val_loss_tp_history,
        'cross entropy' : val_loss_ec_history,
        'accuracy' : val_accuracy_history
    }

    return train_history, val_history
