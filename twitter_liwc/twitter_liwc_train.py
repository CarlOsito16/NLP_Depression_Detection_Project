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
    # scheduler = MultiStepLR(optimizer, milestones = [1000, 2000, 10000], gamma = 0.1)
    progress_bar = tqdm(range(num_training_steps))
    start_time = time.time()
    
    model.train()
    logger = Logger(model_name = 'twitter_liwc', data_name = 'train_1')
    best_val_loss = torch.tensor(float("inf"))
    best_val_acc = torch.tensor(float("-inf"))
    for epoch in range(num_epochs):
        epoch_start = time.time()
        train_acc = 0.0
        train_losses = 0.0
        for i, (batch) in enumerate(train_loader):
            batch_start = time.time()
            model = model.to(device)
            # print(model)
            batch = {k: v.to(device) for k, v in batch.items()}
            batch['labels'] = batch['labels'].squeeze(1)
            batch['input_ids'] = batch['input_ids'].squeeze(1)
            batch['attention_mask'] = batch['attention_mask'].squeeze(1)
            
            optimizer.zero_grad()

            outputs = model(**batch)
            # print(outputs)

            logits = outputs.logits
            # print(logits.size())
            predictions = torch.argmax(logits, dim = 1)
            # targets = torch.argmax(batch['labels'], dim = 1)
            targets = batch["labels"]
            # print("predictions: ", predictions)
            # print("targets: ", targets)
            # print("targets: ", torch.transpose(targets, 0, 1))
            batch_acc = ((predictions == targets).sum()/ len(targets)).item()
            # batch_acc = ((predictions == targets).sum()/ len(targets)).item()

            # print("batch acc", batch_acc)
            # print("corrects", (predictions == targets).size())
            train_acc += batch_acc

            loss = outputs.loss
            loss.backward()
            train_losses+= loss.item()

            # nn.utils.clip_grad_value_(model.parameters(), clip_value=3.0)
            optimizer.step()
            # scheduler.step()
            # print("last LR: ", scheduler.get_last_lr())
            progress_bar.update(1)
            # print(f'Batch Training Time: {batch_end - batch_start} s')
            logger.log(loss, batch_acc, epoch, i, num_batches)

            if i % 100 == 0:
                logger.display_status(epoch, num_epochs, i, num_batches, loss, batch_acc)  
            

        epoch_end = time.time()

        print(f'########### End of Epoch {epoch + 1} ###############')
        print(f'Epochs: {epoch + 1} | Training Batch Loss: {train_losses / len(train_loader): .4f}')
        print(f'Epoch Training Time: {(epoch_end - epoch_start):.4f} s')
        print('')
        print(f'Train Accuracy: {train_acc / len(train_loader): .4f}')
        print(f'Epoch Training Time: {(epoch_end - epoch_start):.4f} s')

        val_accs = 0.0
        val_losses = 0.0

        print("Validation progress ...")
        val_start = time.time()
        with torch.no_grad():
            model.eval()
            for i, (batch) in enumerate(val_loader):
                
                model = model.to(device)
                batch = {k: v.to(device) for k, v in batch.items()}
                batch = {k: v.to(device) for k, v in batch.items()}
                batch['labels'] = batch['labels'].squeeze(1)
                batch['input_ids'] = batch['input_ids'].squeeze(1)
                batch['attention_mask'] = batch['attention_mask'].squeeze(1)
                outputs = model(**batch)

                logits = outputs.logits
                # print(logits.size())
                predictions = torch.argmax(logits, dim = 1)
                # targets = torch.argmax(batch['labels'], dim = 1)
                targets = batch['labels']
                val_acc = ((predictions == targets).sum()/ len(targets)).item()
                val_accs += val_acc

                loss = outputs.loss
                val_losses+= loss.item()
                
                # valiation_progress.update(1)
                val_end = time.time()
            
                if loss.item() < best_val_loss:
                    best_val_loss = loss.item()
                    best_val_acc = val_acc
                    logger.save_models(model, epoch, i)
                # print(f"Validation loss: {loss.item():.4f}")
                # print(f'Validation Accuracy: {acc:.4f}')
            print(f'Total Average Validation Loss: {val_losses/len(val_loader):.4f}')
            print(f'Total Average Validation Accuracy: {val_accs/len(val_loader):.4f}')
            print('')
            print(f'Best Validation Loss: {best_val_loss:.4f}')
            print(f'Best Validation Accuracy: {best_val_acc:.4f}')
            print(f'Validation Time: {(val_end - val_start):.4f} s')
    print('')
    print(f"Total Training Time: {(time.time() - start_time):.4f} s")