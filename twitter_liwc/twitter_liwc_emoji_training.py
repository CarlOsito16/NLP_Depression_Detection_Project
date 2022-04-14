import os
import pandas as pd
from transformers import DataCollatorWithPadding, DistilBertTokenizer
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.data import random_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from twitter_liwc_emoji_dataset import BertDataset
from twitter_liwc_emoji_train import train
from twitter_liwc_emoji_test import test

os.environ['http_proxy'] = 'http://192.41.170.23:3128'
os.environ['https_proxy'] = 'http://192.41.170.23:3128'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

depressed=pd.read_csv('superclean_depressed.csv', delimiter=',', index_col=0).reset_index(drop=True)
controlled=pd.read_csv('superclean_controlled.csv', delimiter=',', index_col=0).reset_index(drop=True)

depressed_75000 = depressed.sample(n=75000, frac=None, replace=False, weights=None, random_state=27, axis=0, ignore_index=False)
controlled_75000 = controlled.sample(n=75000, frac=None, replace=False, weights=None, random_state=27, axis=0, ignore_index=False)

df = pd.concat((depressed_75000, controlled_75000), axis = 0)
print(df.shape)
# df.head()

labels = {'controlled':0,
          'depressed':1
}

print("Loading emoji LIWC Tokenizer")
tokenizer = DistilBertTokenizer.from_pretrained('emoji_liwc_tokenizer', do_lower_case=False, tokenize_chinese_chars= False)
print("Loading Dataset ...")
dataset = BertDataset(df, labels, tokenizer, max_length=64)

### Split
train_size = int(0.8 * df.shape[0])
val_size = int(0.1 * df.shape[0])
test_size = df.shape[0] - (train_size + val_size)

df_train, df_val, df_test = random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(27))

batch_size = 32
data_collator = DataCollatorWithPadding(tokenizer)

train_loader = DataLoader(dataset=df_train,batch_size=batch_size, shuffle = True, num_workers= 4, collate_fn = data_collator)

val_loader = DataLoader(dataset=df_val,batch_size=batch_size, shuffle = True, num_workers= 4, collate_fn = data_collator)

test_loader = DataLoader(dataset=df_test,batch_size=batch_size, shuffle = False, num_workers= 4, collate_fn = data_collator)

print("Loading Pretrained Model ...")
# configuration = DistilBertConfig()
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
model.resize_token_embeddings(len(tokenizer))

print(model)

num_epochs = 5
lr = 2e-5
optimizer = optim.AdamW(model.parameters(), lr = lr)

# print("Start Training ...")
# train(model, train_loader, val_loader, num_epochs, optimizer, device)
# print("Training Ended ...")


print("Loading pretrained weights ...")
model.load_state_dict(torch.load('models/twitter_liwc_emoji/train_1/model_epoch_2_num165'))
print("Start Testing ...")
test(model, test_loader, device)
print("End Testing ..")