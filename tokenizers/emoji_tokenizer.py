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
# from LIWC import *
import advertools as adv


# depressed=pd.read_csv('superclean_depressed.csv', delimiter=',', index_col=0).reset_index(drop=True)
# controlled=pd.read_csv('superclean_controlled.csv', delimiter=',', index_col=0).reset_index(drop=True)

# depressed_75000 = depressed.sample(n=75000, frac=None, replace=False, weights=None, random_state=27, axis=0, ignore_index=False)
# controlled_75000 = controlled.sample(n=75000, frac=None, replace=False, weights=None, random_state=27, axis=0, ignore_index=False)

# df = pd.concat((depressed_75000, controlled_75000), axis = 0)
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=False, tokenize_chinese_chars= False)

# print(df.head())

# emojis = []
# for i, text in enumerate(df['clean_Tweet']):
#     emoji = adv.extract_emoji(text)
#     # print(emoji)

#     emojis.append(emoji['emoji_flat'])
#     if i % 10000 == 0:
#         print(i)

# emo_list = []
# for emos in emojis:
#     if emos:
#         for emo in emos:
#             emo_list.append(emo)

# unique_emo = set(emo_list)

# emoji_list = open("emoji_list.txt", "w")
# for element in unique_emo:
#     emoji_list.write(element + "\n")
# emoji_list.close()
##########################################
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

# tokenizer.save_pretrained('./emoji_tokenizer')
# print("New Emoji Tokenizer Saved ...")
# model.resize_token_embeddings(len(tokenizer))
###########################################