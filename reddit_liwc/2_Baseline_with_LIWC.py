import os
import pandas as pd
from transformers import DataCollatorWithPadding, DistilBertConfig, DistilBertTokenizer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.data import random_split
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from BertDataset import BertDataset
from train import train
from test import test
from LIWC_process import LWIC_Categories_unique_list

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


df = pd.read_csv("reddit_depression_suicide.csv",delimiter=',')
print("df:",df)
print("df.shape : ",df.shape)


labels = {'depression':1,
          'SuicideWatch':0
          }
    
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')


print("Loading Dataset ...")
dataset = BertDataset(df, labels, tokenizer,max_length=64)

### Split
train_size = int(0.8 * df.shape[0])
val_size = int(0.1 * df.shape[0])
test_size = df.shape[0] - (train_size + val_size)

df_train, df_val, df_test = random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(27))



batch_size = 32
data_collator = DataCollatorWithPadding(tokenizer)

train_loader = DataLoader(dataset=df_train,batch_size=batch_size, shuffle = True, num_workers= 4, collate_fn = data_collator)
val_loader = DataLoader(dataset=df_val,batch_size=batch_size, shuffle = True, num_workers= 4, collate_fn = data_collator)
test_loader = DataLoader(dataset=df_test,batch_size=batch_size, shuffle = True, num_workers= 4, collate_fn = data_collator)



LWIC_words = LWIC_Categories_unique_list
tokenizer.add_tokens(LWIC_Categories_unique_list)

print("Loading Pretrained Model ...")
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')


model.resize_token_embeddings(len(tokenizer))

num_epochs = 5
lr = 2e-5
optimizer = optim.AdamW(model.parameters(), lr = lr)



print("Start Training ...")
train(model, train_loader, val_loader, num_epochs, optimizer, device)
print("Training Ended ...")


print("Start Testing ...")
test(model, test_loader, device)
print("End Testing ..")