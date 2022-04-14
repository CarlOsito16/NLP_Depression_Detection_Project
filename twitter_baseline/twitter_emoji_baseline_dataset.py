import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


class BertDataset(Dataset):
    def __init__(self, df, labels, tokenizer, max_length):
        super(BertDataset, self).__init__()
        self.df = df
        # print("loaded df", self.df.shape)
        self.tokenizer = tokenizer
        self.target = self.df.iloc[:,1]
        self.max_length = max_length
        self.labels = [labels[label] for label in self.df['Status']]
        
    def __len__(self):
        return len(self.df)
    
    def get_batch_labels(self, index):
        # Fetch a batch of labels
        _ = np.array(self.labels[index])
        label = torch.tensor(_, dtype = torch.long)
        # one_hot = F.one_hot(_, num_classes = 2)
        # return one_hot.type(torch.float)
        return label.unsqueeze(0)
    
    def __getitem__(self, index):
        # print("index: ", index)
        # print("get item df", self.df.iloc[index, 3])
        text1 = self.df.iloc[index,0]
        # print(text1)
        inputs = self.tokenizer.__call__(
            text1 ,
            None,
            # pad_to_max_length=True,
            add_special_tokens=True,
            return_attention_mask=True,
            max_length=self.max_length,
            truncation=True,
            padding = 'max_length'
        )
        ids = inputs["input_ids"]
        # token_type_ids = inputs["token_type_ids"]
        mask = inputs["attention_mask"]
        # print("###########")
        # print("ids: ", ids)
        # print("###########")
        return {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'attention_mask': torch.tensor(mask, dtype=torch.long),
            # 'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            # 'labels': F.one_hot(torch.tensor(self.get_batch_labels(index), dtype=torch.long), num_classes = 2)
            'labels' : self.get_batch_labels(index)
        }