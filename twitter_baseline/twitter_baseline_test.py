import os
import time
import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


os.environ['http_proxy'] = 'http://192.41.170.23:3128'
os.environ['https_proxy'] = 'http://192.41.170.23:3128'

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
            # print(logits.size())
            predictions = torch.argmax(logits, dim = 1)
            # targets = torch.argmax(batch['labels'], dim = 1)
            targets = batch['labels']
            test_acc = ((predictions == torch.transpose(targets, 0, 1)).sum()/ len(targets)).item()
            test_accs += test_acc

            loss = outputs.loss
            test_losses+= loss.item()
            
            # valiation_progress.update(1)
            test_end = time.time()
        
        print(f'Total Average Testing Loss: {test_losses/len(test_loader):.4f}')
        print(f'Total Average Testing Accuracy: {test_accs/len(test_loader):.4f}')
        print('')
        print(f'Testing Time: {(test_end - test_start):.4f} s')