import torch
import time as t
import numpy as np

from model import PADDING_CONST

def run_epoch(model, dataloader, device, optimizer=None):
    
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

def train(model, n_epochs, optimizer, train_loader, val_loader, scheduler=None, device=None, verbose=True, freq=None):
    if verbose and freq is None:
        freq = max(n_epochs // 10, 1)
    if device is None:
        device = GLOBAL_DEVICE

    train_loss_ll_history, train_loss_tp_history, train_loss_ec_history, train_accuracy_history = [], [], [], []
    val_loss_ll_history, val_loss_tp_history, val_loss_ec_history, val_accuracy_history = [], [], [], []

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

        if scheduler is not None:
            scheduler.step()
        
        time_epoch_end = t.time() - time_start

        if verbose and epoch % freq == 0:
            print("[ Epoch {} ]".format(epoch))
            print("(Training)     log-likelihood: {}, RMSE: {}, CE: {}, accuracy: {}".format( train_loss_ll, np.sqrt(train_loss_tp), train_loss_ec, train_accuracy ))
            print("(Validation)   log-likelihood: {}, RMSE: {}, CE: {}, accuracy: {}".format( val_loss_ll, np.sqrt(val_loss_tp), val_loss_ec, val_accuracy ))
            print("Time elapsed: {:.2f} s".format(time_epoch_end))

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