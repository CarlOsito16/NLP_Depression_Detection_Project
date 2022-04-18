import os
import time
import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def test(model, test_loader, device):
    num_testing_steps = len(test_loader)
    progress_bar = tqdm(range(num_testing_steps))
    test_start = time.time()
    with torch.no_grad():
        test_losses = 0.0
        test_accs = 0.0
        model.eval()
        for i, (batch) in enumerate(test_loader):
            
            model = model.to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)

            logits = outputs.logits
            
            predictions = torch.argmax(logits, dim = 1)
           
            targets = batch['labels']
            test_acc = ((predictions == torch.transpose(targets, 0, 1)).sum()/ len(targets)).item()
            test_accs += test_acc

            loss = outputs.loss
            test_losses+= loss.item()
            
         
            test_end = time.time()
        
        print(f'Total Average Testing Loss: {test_losses/len(test_loader):.4f}')
        print(f'Total Average Testing Accuracy: {test_accs/len(test_loader):.4f}')
        print('')
        print(f'Validation Time: {(test_end - test_start):.4f} s')


