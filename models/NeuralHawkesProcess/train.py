import time 
import torch
from sklearn.metrics import accuracy_score

def run_epoch(model, optimizer, dataloader, device, mode = 'train', sum_losses=True, scale=0.001):
    if mode == 'train':
      model.train(True)
    else:
      model.train(False)

    is_train = (mode == 'train')
    epoch_loss, event_num = 0, 0
    epoch_llh, epoch_time_error, epoch_event_error, event_num, epoch_event_acc = 0, 0, 0, 0, 0
    with torch.set_grad_enabled(is_train):        
        for event_seq, time_seq in dataloader:

            event_seq, time_seq = event_seq.to(device), time_seq.to(device)
            intens, time, event = model.forward(event_seq, time_seq)

            # Log-likelihood (loss for the whole sequence)
            loss_llh = model.LogLikelihoodLoss(intens, time_seq)

            if sum_losses:
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
                
            if is_train:
                optimizer.zero_grad()
                loss.backward() 
                optimizer.step()

            epoch_llh += loss_llh.detach().cpu().numpy()
            event_num += time_seq.shape[0] * time_seq.shape[1]

    if sum_losses:
        return epoch_llh/event_num, epoch_time_error/len(dataloader), epoch_event_error/len(dataloader), epoch_event_acc/len(dataloader)
    else:
        return epoch_llh/event_num


def train(model, optimizer, train_loader, val_loader, device, n_epochs=50, sum_losses=True, save_path=None, verbose_epoch=1):
    
    
    start = time.time()
    best_loss = 10e15
    statistics = {'train': [], 'val': []}
    for epoch in range(n_epochs):

        train_stats = run_epoch(model, optimizer, train_loader, device, mode = 'train',sum_losses=sum_losses)
        val_stats = run_epoch(model, None, val_loader, device, mode = 'val',sum_losses=sum_losses)

        train_val_time = time.time() - start 
        statistics['train'].append(train_stats)
        statistics['val'].append(val_stats)

        if save_path:
            if statistics['val'][-1][0] < best_loss:
                best_loss = statistics['val'][-1][0]
                torch.save(model.state_dict(), save_path)

        if epoch % verbose_epoch == 0:
            print('Epoch:', epoch)
            if sum_losses:
                print('Log-Likelihood:: train:', -statistics['train'][-1][0], ', val:', -statistics['val'][-1][0])
                print('Time MSE:: train:', statistics['train'][-1][1], ', val:', statistics['val'][-1][1])
                print('Event CE:: train:', statistics['train'][-1][2], ', val:', statistics['val'][-1][2])    
                print('Event pred accuracy:: train:', statistics['train'][-1][3], ', val:', statistics['val'][-1][3])    
            else:
                print('Log-Likelihood:: train:', -statistics['train'][-1], ', val:', -statistics['val'][-1])
            print('time:', train_val_time)
            print('-'*60)

    return statiscs
