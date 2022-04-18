import time
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
from logger import Logger

import warnings

warnings.filterwarnings('ignore')

def train(model, train_loader, val_loader, num_epochs, optimizer, device):
    num_training_steps = num_epochs * len(train_loader)
    num_batches = len(train_loader)
    progress_bar = tqdm(range(num_training_steps))
    start_time = time.time()
    
    model.train()
    logger = Logger(model_name = 'distilbert-base-uncased', data_name = 'Reddit_Train_')
    for epoch in range(num_epochs):
        epoch_start = time.time()
        train_acc = 0.0
        train_losses = 0.0
        for i, (batch) in enumerate(train_loader):
            batch_start = time.time()
            model = model.to(device)
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()

            outputs = model(**batch)
            # print(outputs)

            logits = outputs.logits
            # print(logits.size())
            predictions = torch.argmax(logits, dim = 1)
            targets = batch["labels"]
            batch_acc = ((predictions == torch.transpose(targets, 0, 1)).sum()/ len(targets)).item()

            train_acc += batch_acc

            loss = outputs.loss
            loss.backward()
            train_losses+= loss.item()

            optimizer.step()
            progress_bar.update(1)
            logger.log(loss, batch_acc, epoch, i, num_batches)

            if i % 100 == 0:
                logger.display_status(epoch, num_epochs, i, num_batches, loss, batch_acc)  
        epoch_end = time.time()

        print(
            f' ########### End of Epoch {epoch + 1} ############### \
            | Epochs: {epoch + 1} | Training Batch Loss: {train_losses / len(train_loader): .4f} \
            | Epoch Training Time: {epoch_end - epoch_start} s \
            | Train Accuracy: {train_acc / len(train_loader): .4f} \
            | Epoch Training Time: {epoch_end - epoch_start} s')

        val_accs = 0.0
        val_losses = 0.0
        best_val_loss = torch.tensor(float("inf"))
        best_val_acc = 0.0
       
        print("Validation progress ...")
        val_start = time.time()
        with torch.no_grad():
            model.eval()
            for i, (batch) in enumerate(val_loader):
                
                model = model.to(device)
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)

                logits = outputs.logits
               
                predictions = torch.argmax(logits, dim = 1)
                
                targets = batch['labels']
                val_acc = ((predictions == torch.transpose(targets, 0, 1)).sum()/ len(targets)).item()
                val_accs += val_acc

                loss = outputs.loss
                val_losses+= loss.item()
            
                val_end = time.time()
            
                if loss.item() < best_val_loss:
                    best_val_loss = loss.item()
                    best_val_acc = val_acc
                    logger.save_models(model, epoch, i)
              
            print(f'Total Average Validation Loss: {val_losses/len(val_loader)}')
            print(f'Total Average Validation Accuracy: {val_accs/len(val_loader)}')
            print(f'Best Validation Loss: {best_val_loss}')
            print(f'Best Validation Accuracy: {best_val_acc}')
            print(f'Validation Time: {val_end - val_start} s')

    print(f"Total Training Time: {time.time() - start_time} s")