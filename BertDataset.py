import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


class BertDataset(Dataset):
    def __init__(self, df, labels, tokenizer, max_length):
        super(BertDataset, self).__init__()
        self.df = df
        print("loaded df", self.df.shape)
        self.tokenizer = tokenizer
        #self.target = df.iloc[:,1]
        self.target = df.iloc[:,0]
        self.max_length = max_length
        self.labels = [labels[label] for label in self.df['label']]
        
    def __len__(self):
        return len(self.df)
    
    def get_batch_labels(self, index):
        # Fetch a batch of labels
        _ = np.array(self.labels[index])
        label = torch.tensor(_, dtype = torch.long)
        return label.unsqueeze(0)
    
    def __getitem__(self, index):
        text1 = self.df.iloc[index,0]
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
        mask = inputs["attention_mask"]
        return {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'attention_mask': torch.tensor(mask, dtype=torch.long),
            'labels' : self.get_batch_labels(index)
        }


