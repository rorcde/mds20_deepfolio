import time 
import torch
from sklearn.metrics import accuracy_score

def run_epoch(model, optimizer, dataloader, device, scale=0.001):
    '''
    Runs one epoch of training.

    Input:
        model - NHP model,
        dataloader - data loader,
        device - current device,
        optimizer - optimizer to use (if None, then validation run is performed),
        scale (optional) - scale factor for time_loss (if None, then only LogLike_loss will be used)
    Output:
        epoch_llh - log-likelihood loss value,
        epoch_time_error - mean squared error of time prediction,
        epoch_event_error - event prediction cross entropy loss value,
        epoch_event_acc - event prediction accuracy value,
        (if scale is None, then only epoch_llh will be returned)
    '''

    epoch_loss, event_num = 0, 0
    epoch_llh, epoch_time_error, epoch_event_error, event_num, epoch_event_acc = 0, 0, 0, 0, 0
    with torch.set_grad_enabled(optimizer is not None):        
        for event_seq, time_seq in dataloader:

            # make forward pass
            event_seq, time_seq = event_seq.to(device), time_seq.to(device)
            intens, time, event = model.forward(event_seq, time_seq)

            # Log-likelihood (loss for the whole sequence)
            loss_llh = model.LogLikelihoodLoss(intens, time_seq)

            if scale is not None:
                # time and type prediction losses
                loss_time = model.time_loss(time, time_seq)
                loss_event = model.event_loss(event, event_seq)
                loss = loss_llh + loss_time + scale * loss_event

                # log results
                epoch_event_error += loss_event.detach().cpu().numpy()
                epoch_time_error += loss_time.detach().cpu().numpy()
                epoch_event_acc += accuracy_score(event[:,:-1].argmax(dim=2).cpu().reshape(-1), 
                                                  event_seq[:, 1:].cpu().reshape(-1))

            else:
                loss = loss_llh
                
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward() 
                optimizer.step()

            epoch_llh += loss_llh.detach().cpu().numpy()
            event_num += time_seq.shape[0] * time_seq.shape[1]

    if scale is not None:
        return epoch_llh/event_num, epoch_time_error/len(dataloader), epoch_event_error/len(dataloader), epoch_event_acc/len(dataloader)
    else:
        return epoch_llh/event_num


def train(model, optimizer, train_loader, val_loader, device, scheduler=None, n_epochs=50, scale=0.001, save_path=None, verbose_epoch=1):
    
    '''
    Trains NHP model.

    Input:
        model - NHP model,
        optimizer - optimizer to use (if None, then validation run is performed),
        train_loader - train dataloader,
        val_loader - val dataloader,
        device - current device,
        scheduler - learning rate scheduler,
        n_epochs (int) - number of epochs for training,
        scale (optional) - scale factor for time_loss (if None, then only LogLike_loss will be used),
        save_path (str) - path to save model with best val loss score,
        verbose_epoch (int) - number of epochs to print intermediate results
    Output:
        statistics (dict) - dictionary with all train and validation statistics
    '''
    
    start = time.time()
    best_loss = 10e15
    statistics = {'train': [], 'val': []}
    for epoch in range(n_epochs):

        train_stats = run_epoch(model, optimizer, train_loader, device, scale=scale)
        val_stats = run_epoch(model, None, val_loader, device, scale=scale)

        train_val_time = time.time() - start 
        statistics['train'].append(train_stats)
        statistics['val'].append(val_stats)


        if scheduler is not None:
            if type(scheduler).__name__ != 'ReduceLROnPlateau':
                scheduler.step()
            else:
                criterion_value = -statistics['val'][-1][0]+scale*statistics['val'][-1][1]+statistics['val'][-1][2] if scale is not None else -statistics['val'][-1]
                scheduler.step(criterion_value)

        if save_path:
            if statistics['val'][-1][0] < best_loss:
                best_loss = statistics['val'][-1][0]
                torch.save(model.state_dict(), save_path)

        if epoch % verbose_epoch == 0:
            print('Epoch:', epoch)
            if scale is not None:
                print('Log-Likelihood:: train:', -statistics['train'][-1][0], ', val:', -statistics['val'][-1][0])
                print('Time MSE:: train:', statistics['train'][-1][1], ', val:', statistics['val'][-1][1])
                print('Event CE:: train:', statistics['train'][-1][2], ', val:', statistics['val'][-1][2])    
                print('Event pred accuracy:: train:', statistics['train'][-1][3], ', val:', statistics['val'][-1][3])    
            else:
                print('Log-Likelihood:: train:', -statistics['train'][-1], ', val:', -statistics['val'][-1])
            print('time:', train_val_time)
            print('-'*60)

    return statistics
