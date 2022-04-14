import os
import pandas as pd
import math
import time
import numpy as np
from transformers import DataCollatorWithPadding, AdamW
import random
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from transformers import AutoModelForMaskedLM, DistilBertTokenizer
from MLM import BertDataset
from tqdm import tqdm
from logger import Logger

os.environ['http_proxy'] = 'http://192.41.170.23:3128'
os.environ['https_proxy'] = 'http://192.41.170.23:3128'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

depressed=pd.read_csv('superclean_depressed.csv', delimiter=',', index_col=0).reset_index(drop=True)
controlled=pd.read_csv('superclean_controlled.csv', delimiter=',', index_col=0).reset_index(drop=True)

depressed_75000 = depressed.sample(n=75000, frac=None, replace=False, weights=None, random_state=27, axis=0, ignore_index=False)
controlled_75000 = controlled.sample(n=75000, frac=None, replace=False, weights=None, random_state=27, axis=0, ignore_index=False)

depressed_not75000 = depressed.drop(depressed_75000.index)
controlled_not75000 = controlled.drop(controlled_75000.index)

df = pd.concat((depressed_not75000, controlled_not75000), axis = 0)
# print(df.shape)

print("Loading LIWC Tokenizer")
tokenizer = DistilBertTokenizer.from_pretrained('liwc_tokenizer', do_lower_case=False, tokenize_chinese_chars= False)

dataset = BertDataset(df, tokenizer, max_length=64)

### Split
train_size = int(0.9 * df.shape[0])
val_size = int(0.1 * df.shape[0])
# test_size = df.shape[0] - (train_size + val_size)

df_train, df_val = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(27))

batch_size = 32
data_collator = DataCollatorWithPadding(tokenizer)

train_loader = DataLoader(dataset=df_train, batch_size=batch_size, shuffle = True, num_workers= 4, collate_fn = data_collator)

val_loader = DataLoader(dataset=df_val, batch_size=batch_size, shuffle = True, num_workers= 4, collate_fn = data_collator)

# test_loader = DataLoader(dataset=df_test,batch_size=batch_size, shuffle = False, num_workers= 4, collate_fn = data_collator)
# print(len(train_loader))
# for i, batch in enumerate(train_loader):
#     print(batch)
#     break

print("Length of dataset", len(dataset))
print("Length of train_loader", len(train_loader))
print("Length of val_loader", len(val_loader))


model_checkpoint = 'distilbert-base-uncased'
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
model.resize_token_embeddings(len(tokenizer))

# print(model)
model = model.to(device)
optimizer = AdamW(model.parameters(), lr = 1e-6, weight_decay = 0.01)

logger = Logger(model_name = 'MLM', data_name = 'train_1')

print("Training Starting ...")
model.train()

epochs = 2
for epoch in range(epochs):
    loop = tqdm(train_loader, leave = True)
    for i, batch in enumerate(loop):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].squeeze(1).to(device)
        attention_mask = batch['attention_mask'].squeeze(1).to(device)
        labels = batch['labels'].squeeze(1).to(device)
        # print("input_ids: ", input_ids.shape)
        # print("input_ids: ", labels.size())
        # print("input_ids: ", attention_mask.size())

        outputs = model(input_ids = input_ids, attention_mask = attention_mask, labels = labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        logger.log(loss, batch_acc = 0, epoch = epoch, n_batch = i, num_batches = len(train_loader))
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss = loss.item())

    best_perplexity = float("inf")
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
        
            model.eval()
            input_ids = batch['input_ids'].squeeze(1).to(device)
            attention_mask = batch['attention_mask'].squeeze(1).to(device)
            labels = batch['labels'].squeeze(1).to(device)

            outputs = model(input_ids = input_ids, attention_mask = attention_mask, labels = labels)
            loss = outputs.loss
            perplexity = math.exp(loss.item())

            if perplexity < best_perplexity:
                print(f"Updating Perplexity to best perplexity of {perplexity}")
                best_perplexity = perplexity
    print(f"Saving the Model ...")
    torch.save(model.state_dict(),
        '{}/model_epoch_{}'.format("./models/MLM", epoch))
