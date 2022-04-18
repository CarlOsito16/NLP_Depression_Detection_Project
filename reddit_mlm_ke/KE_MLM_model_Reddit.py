from transformers import AutoModelForMaskedLM, AutoTokenizer
from LIWC_process import LWIC_Categories_unique_list
import numpy as np
import pandas as pd
from torch import nn
from transformers import BertModel



model_checkpoint = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)


depression_related_token_dict = dict()

for word in LWIC_Categories_unique_list:
    result = tokenizer.tokenize(word)
    id = tokenizer.convert_tokens_to_ids(result)
    num_token = len(result)

    for token in range(num_token):
        if id[token] in list(depression_related_token_dict.values()):
            pass
        else:
            depression_related_token_dict[result[token]] = id[token]


def search_token_name(token_ids):
    for name, ids in depression_related_token_dict.items():
        if ids == token_ids:
            return name



dataframe = pd.read_csv("reddit_depression_suicide.csv")
df_train, df_val, df_test = np.split(dataframe.sample(frac=1, random_state=42),
         [int(.8*len(dataframe)), int(.9*len(dataframe))] )


print(len(df_train))
print(len(df_val))
print(len(df_test))


def make_tokenize(sample):
  sentences = [sen for sen in sample]
  sentence_sample = len(sentences)
  # print(sentences)
  # print(sentence_sample)
  inputs = tokenizer(sentences, return_tensors = 'pt', truncation=True, padding=True)
  # print(len(inputs))
  return inputs,sentence_sample


train_inputs = make_tokenize(df_train)[0]
val_inputs = make_tokenize(df_val)[0]
test_inputs = make_tokenize(df_test)[0]


import math
import random
import torch

def search_token_name(token_ids):
    for name, ids in depression_related_token_dict.items():
        if ids == token_ids:
            return name


def selective_masking(inputs):
  sentence_sample = make_tokenize(df_train)[1]

  for sentence in range(0, sentence_sample):
    inputs['labels'] = inputs.input_ids.detach().clone()
    input_token = inputs.input_ids[sentence].tolist()
    # print("sentences:",sentences[sentence])
    # print("input_token:",input_token)



    to_mask_token_id = []
    for token in input_token:
      if token in list(depression_related_token_dict.values()):
        print(token, search_token_name(token))
        to_mask_token_id.append(token)
      else:
        pass
    to_mask_token_id = list(set(to_mask_token_id))

    # print("to_mask_token_id:",to_mask_token_id)


    random_pop = 0.15 
    number_mask_token = math.ceil(random_pop * len(to_mask_token_id))
    random.seed(10)
    random_number_mask_token = random.sample(to_mask_token_id, number_mask_token)

    # print("number_mask_token:",number_mask_token)
    # print("random_number_mask_token:",random_number_mask_token)

    for random_token in random_number_mask_token:
      index = inputs.input_ids[sentence].tolist().index(random_token)
      # print("random_token,index:",random_token, index)

      inputs.input_ids[sentence, index ] = 103
      # print("inputs.input_ids[sentence]:",inputs.input_ids[sentence])
    return inputs
    # break

    # print()

train_inputs_news = selective_masking(train_inputs)


class DepressionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)

train_dataset = DepressionDataset(train_inputs_news)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = 3, shuffle = True)

device = torch.device('cuda:3') if torch.cuda.is_available() else torch.device('cpu')


from transformers import AdamW
from tqdm import tqdm


epochs = 5
learning_rate = 0.00001
optim = torch.optim.AdamW(model.parameters(), lr = learning_rate)
model.to(device)


for epoch in range(epochs):

  total_acc_train = 0
  total_loss_train = 0



  loop = tqdm(train_dataloader, leave =True)
  for batch in loop:
    optim.zero_grad()
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)

    outputs = model(input_ids, attention_mask = attention_mask, labels =labels)
    
    loss = outputs.loss

    loss.backward()
    optim.step()

    #torch.save(model.state_dict(), f'KE_MLM_model_Reddit.pt')

    loop.set_description(f"Epoch: {epoch}")
    loop.set_postfix(loss = loss.item())


  PATH = '/NLP_drepression/Depress'
  model.save_pretrained("PATH_Reddit")
  model.state_dict()


