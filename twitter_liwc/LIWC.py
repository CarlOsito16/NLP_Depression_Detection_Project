import pandas as pd
import time
import math
import random
import numpy as np
from transformers import AutoModelForMaskedLM, DistilBertTokenizer, PreTrainedTokenizerFast
import json
from collections import OrderedDict


# model_checkpoint = 'distilbert-base-uncased'
# model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
# print(model)


print("Loading LIWC Tokenizer")
tokenizer = DistilBertTokenizer.from_pretrained('liwc_tokenizer', do_lower_case=False, tokenize_chinese_chars= False)

# tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=False, tokenize_chinese_chars= False)

liwc = pd.read_csv('LIWC_process_new.csv')

depression_topic_dic = OrderedDict({(122, "family"), (127, 'negative_emo'), (128, 'anxiety'),
                 (148, 'health'), (359, 'religion'), (360, 'death') , 
                (22, 'swear') })

family_word = []
netative_emo_word =[]
anxiety_word = []
health_word = []
religion_word = []
death_word = []

word_matrix = np.zeros((len(liwc), len(list(depression_topic_dic.keys()))))

for idx in range(len(liwc)):
    word = liwc.iloc[idx]['clean_word']
    cat = json.loads(liwc['cat_ids'].iloc[idx])
    # print((idx, word))
    for subcat in cat:
        if subcat in list(depression_topic_dic.keys()):
            word_matrix[idx, list(depression_topic_dic.keys()).index(subcat) ] = 1

new_liwc = liwc.join(pd.DataFrame(word_matrix, columns =list(depression_topic_dic.values()), dtype = int ))

anxiety_word = new_liwc[new_liwc['anxiety'] == 1]['clean_word'].tolist()
swear_word = new_liwc[new_liwc['swear'] == 1]['clean_word'].tolist()
death_word = new_liwc[new_liwc['death'] == 1]['clean_word'].tolist()
negative_emo_word = new_liwc[new_liwc['negative_emo'] == 1]['clean_word'].tolist()
family_word = new_liwc[new_liwc['family'] == 1]['clean_word'].tolist()
religion_word = new_liwc[new_liwc['religion'] == 1]['clean_word'].tolist()
health_word = new_liwc[new_liwc['health'] == 1]['clean_word'].tolist()

depression_related_cat_name = ["anxiety_word", "swear_word", "death_word", "negative_emo_word", 
                          "family_word", "religion_word",  "health_word"  ]

depression_related_cat = [anxiety_word, swear_word, death_word, negative_emo_word, 
                          family_word, religion_word,  health_word  ]

cumulative_count = []
uniqe_cumulative_count = []
list_of_word = []
for cat_idx in range(0, 7):
    list_of_word.extend(depression_related_cat[cat_idx])
    cumulative_count.append(len(list_of_word))
    uniqe_cumulative_count.append(len(set(list_of_word)))

unique_depressed_word_list = list(set(list_of_word))
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