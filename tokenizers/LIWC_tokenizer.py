import os
import pandas as pd
import numpy as np
from transformers import DistilBertTokenizer, DistilBertTokenizer
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.data import random_split
from twitter_baseline_dataset import BertDataset
import warnings
warnings.filterwarnings('ignore')
from LIWC import *

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=False, tokenize_chinese_chars= False)

depression_related_token_dict = dict()

for word in unique_depressed_word_list:
    result = tokenizer.tokenize(word)
    id = tokenizer.convert_tokens_to_ids(result)
    num_token = len(result)

    for token in range(num_token):
        if id[token] in list(depression_related_token_dict.values()):
            pass
        else:
            depression_related_token_dict[result[token]] = id[token]

print("length of depression tokens from tokenizer:" ,len(depression_related_token_dict))

# tokenizer = tokenizer.from_pretrained()
print("before tokenizer added", len(tokenizer.get_vocab()))

liwc = pd.read_csv('LIWC_process_new.csv')
# print(liwc.head())
unique_liwc = np.unique(liwc['clean_word']).tolist()
# print(type(unique_liwc), len(unique_liwc))

# print(len(tokens))
# print(unique_depressed_word_list)


tokenizer.add_tokens(unique_liwc)
print("len of tokenized tokenizer after added", len(tokenizer.get_vocab()))

print("added tokens", tokenizer.get_added_vocab())

# tokenizer.save_pretrained('./liwc_tokenizer')
# print("New Tokenizer Saved ...")
# model.resize_token_embeddings(len(tokenizer))