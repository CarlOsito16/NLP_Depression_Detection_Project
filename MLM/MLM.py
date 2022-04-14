
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import random
import math
import json
from collections import OrderedDict
from LIWC import *


class BertDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        super(BertDataset, self).__init__()
        self.df = df
        # print("loaded df", self.df.shape)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, indx):
        text1 = self.df.iloc[indx,0]
        inputs = self.tokenizer.__call__(
            text1 ,
            None,
            add_special_tokens=True,
            return_attention_mask=True,
            max_length=self.max_length,
            truncation = True,
            padding = 'max_length',
            return_tensors = "pt"
        )
        mask = inputs["attention_mask"]
        ######### Masking 15% of depression tokens in a sentence ##################################################
        inputs['labels'] = inputs.input_ids.detach().clone()
        input_token = inputs.input_ids[0].tolist()
        to_mask_token_id = []

        for token in input_token:
            if token in list(depression_related_token_dict.values()):
                # print(token, search_token_name(token))
                to_mask_token_id.append(token)
            else:
                pass

        to_mask_token_id = list(set(to_mask_token_id))

        # print("to_mask_token_id: ", to_mask_token_id)

        random_pop = 0.15
        number_mask_token = math.ceil(random_pop * len(to_mask_token_id))
        random.seed(10)
        random_number_mask_token = random.sample(to_mask_token_id, number_mask_token)

        # print("number_mask_token: ", number_mask_token)
        # print("random_number_mask_token: ", random_number_mask_token)


        for random_token in random_number_mask_token:
            index = inputs.input_ids[0].tolist().index(random_token)
            # print("random_token, index: ", random_token, index)

            inputs.input_ids[0, index] = 103
        # print("ids: ", inputs['input_ids'])
        # raise ValueError("Testing Ended")
        ############ 15% chance to mask tokens which include depression words ############################################
        # ids = inputs.input_ids.detach().clone()
        # inputs["labels"] = inputs.input_ids.detach().clone()
        # mask_arr = (ids != 0) * (ids != 101) * (ids != 102)
        # # print("mask_arr: ", mask_arr)
        # depression = list(depression_related_token_dict.values())
        # seq_len = torch.flatten(mask_arr[0].nonzero())
        # rand = torch.rand(len(seq_len))
        # prob = rand < 0.15
        # mask_arr[0, 1:len(prob)+1] = prob
        # mask_seq = torch.flatten(mask_arr[0].nonzero()).tolist()
        # to_mask = ids[0, mask_seq]
        # included = any(i for i in to_mask if i in depression)
        # i = 0
        # while not included and i < 3:
        #     rand = torch.rand(len(seq_len))
        #     prob = rand < 0.15
        #     mask_arr[0, 1:len(prob)+1] = prob
        #     mask_seq = torch.flatten(mask_arr[0].nonzero()).tolist()
        #     to_mask = ids[0, mask_seq]
        #     included = any(i for i in to_mask if i in depression)
        #     i += 1
        #     # print(i)
        # # print("check whether in or not", included)
        # # print("before mask: ", to_mask)
        # # to_mask = [(lambda x: 103)(x) for x in to_mask]
        # # print("type is: ", (torch.tensor([(lambda x: 103)(x) for x in to_mask], dtype = torch.DoubleTensor)).dtype)
        # # print("ids dtype", ids.dtype)
        # ids[0, mask_seq] = torch.tensor([(lambda x: 103)(x) for x in to_mask], dtype = torch.int64)
        
        # print("ids: ", ids.dtype)
        # print("labels: ", inputs['labels'].dtype)
        # print("mask: ", mask.dtype)
        # raise ValueError("Testing Finished")
    ###################################################################
        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': mask,
            'labels' : inputs['labels']}