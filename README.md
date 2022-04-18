# NLP_Depression_Detection_Project


- "datasets" folder contains superclean_controlled.csv and superclean_depressed.csv, which are already cleaned, for twitter and reddit_depression_suicide.csv which are from reddit.

- "MLM" folder is for KE_MLM model processing and to save the model.

- LIWC tokens processing are saved in "tokenizers".

- "reddit_baseline", "reddit_liwc" and "reddit_mlm_ke" are for the baseline Distilbert_base_uncased model , model with added LIWC tokens and knowledge-enhanced model with masking respectively.

- "twitter_baseline", "twitter_liwc" and "twittwe_mlm_ke" are the same with reddit's part.

- "runs" is for the saved log and "weights" is our trained weights.

_ "BertDataset.py" is for customDataset class and "logger.py" for tensorboard things.

- "PreprocessingCombined.ipynb" is for the data preprocessing(http removal, non-english word removal, etc)
