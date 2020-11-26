# -*- coding: utf-8 -*-

import time 
import torch

def BeginningOfStream(batch_data, type_size):
            """
              While initializing LSTM we have it read a special beginning-of-stream (BOS) event (k0, t0), 
              where k0 is a special event type and t0 is set to be 0 
              (expanding the LSTM’s input dimensionality by one) see Appendix A.2
            """

            seq_events, seq_time, seq_tot_time, seqs_len = batch_data

            pad_event = torch.zeros_like(seq_events[:,0]) + type_size
            pad_time = torch.zeros_like(seq_time[:,0])
            pad_event_seqs = torch.cat((pad_event.reshape(-1,1), seq_events), dim=1)
            pad_time_seqs = torch.cat((pad_time.reshape(-1,1), seq_time), dim=1)

            return pad_event_seqs.long(), pad_time_seqs, seq_tot_time, seqs_len

def run_epoch(model, optimizer, criterion, dataloader, device, mode = 'train'):
    if mode == 'train':
      model.train(True)
    else:
      model.train(False)

    is_train = (mode == 'train')
    epoch_loss, event_num = 0, 0
    with torch.set_grad_enabled(is_train):
        for sample in dataloader:

            event_seqs, time_seqs, total_time_seqs, seqs_length = BeginningOfStream(sample, model.type_size)
            output = model.forward(event_seqs.to(device), time_seqs.to(device))
            loss = criterion(model, event_seqs.to(device), time_seqs, seqs_length,
                                              total_time_seqs.to(device), output)

            if is_train:
                optimizer.zero_grad()
                loss.backward() 
                optimizer.step()

            batch_event_num = torch.sum(seqs_length)
            epoch_loss += loss.detach().cpu().numpy()
            event_num += batch_event_num

    return epoch_loss/event_num


def train(model, optimizer, criterion, train_loader, val_loader, device, n_epochs=50, eval_pred=False, save_path=None):
    
    
    start = time.time()
    best_loss = 10e15
    loss_history = {'train': [], 'val': []}
    time_mse_error_history = {'train': [], 'val': []}
    type_accuracy_history = {'train': [], 'val': []}
    for epoch in range(n_epochs):

        train_loss = run_epoch(model, optimizer, criterion, train_loader, device, mode = 'train')
        val_loss = run_epoch(model, None, criterion, val_loader, device, mode = 'val')

        if eval_pred:
          #time_mse_error_train, type_accuracy_train = evaluate_prediction(model, train_loader)
          time_mse_error_val, type_accuracy_val = evaluate_prediction(model, val_loader, device)
        train_val_time = time.time() - start
        
        loss_history['train'].append(train_loss)
        loss_history['val'].append(val_loss)

        #time_mse_error_train_history['train'].append(time_mse_error_train)
       # time_mse_error_history['val'].append(time_mse_error_val)

        #type_accuracy_train_history['train'].append(type_accuracy_train)
        #type_accuracy_history['val'].append(type_accuracy_val)

        if val_loss < best_loss:
            best_loss = val_loss
            if save_path:
                torch.save(model.state_dict(), save_path)

        if epoch % 1 == 0:
            print('Epoch:', epoch)
            print('train_log_likelihood:', -train_loss, 'val_log_likelihood', -val_loss)

            if eval_pred:
                print('time_mse_error_val:', time_mse_error_val, 'type_accuracy_val:', type_accuracy_val)

            print('time:', train_val_time)
            print('-'*60)

    return loss_history, time_mse_error_history, type_accuracy_history