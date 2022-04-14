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
import advertools as adv


print("Loading LIWC Tokenizer")
tokenizer = DistilBertTokenizer.from_pretrained('liwc_tokenizer', do_lower_case=False, tokenize_chinese_chars= False)

emoji_list = []
with open('emoji_list.txt') as fp:
    contents = fp.read()
    for i, entry in enumerate(contents):
        if i % 2 == 0:
            emoji_list.append(entry)

# print(emoji_list)
# print(len(emoji_list))
emoji_dict = dict()

for emoji in emoji_list:
    result = tokenizer.tokenize(emoji)
    id = tokenizer.convert_tokens_to_ids(result)
    num_token = len(result)

    for token in range(num_token):
        if id[token] in list(emoji_dict.values()):
            pass
        else:
            emoji_dict[result[token]] = id[token]

print("length of emoji tokens from tokenizer:" ,len(emoji_dict))

print("before tokenizer added", len(tokenizer.get_vocab()))

print("length of unique emoji", len(emoji_list))

tokenizer.add_tokens(emoji_list)
print("len of tokenized tokenizer after added", len(tokenizer.get_vocab()))

print("added tokens", tokenizer.get_added_vocab())

# tokenizer.save_pretrained('./emoji_liwc_tokenizer')
# print("New Emoji Tokenizer Saved ...")
# model.resize_token_embeddings(len(tokenizer))
###########################################