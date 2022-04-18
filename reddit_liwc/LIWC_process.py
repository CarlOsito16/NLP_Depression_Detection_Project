import numpy as np
import pandas as pd


LWIC_words = pd.read_excel("LWIC_collection.xlsx" )

swear_word = LWIC_words[(LWIC_words.Ref1 == 22) | (LWIC_words.Ref2 == 22) | (LWIC_words.Ref3 == 22)
                        | (LWIC_words.Ref4 == 22) | (LWIC_words.Ref5 == 22) | (LWIC_words.Ref6 == 22)
                        | (LWIC_words.Ref7 == 22) | (LWIC_words.Ref8 == 22)]
clean_swear_word = swear_word['Word'].str.replace('*', '')



family_word = LWIC_words[(LWIC_words.Ref1 == 122) | (LWIC_words.Ref2 == 122) | (LWIC_words.Ref3 == 122)
                        | (LWIC_words.Ref4 == 122) | (LWIC_words.Ref5 == 122) | (LWIC_words.Ref6 == 122)
                        | (LWIC_words.Ref7 == 122) | (LWIC_words.Ref8 == 122)]
clean_family_word = family_word['Word'].str.replace('*', '')



negemo_word = LWIC_words[(LWIC_words.Ref1 == 127) | (LWIC_words.Ref2 == 127) | (LWIC_words.Ref3 == 127)
                        | (LWIC_words.Ref4 == 127) | (LWIC_words.Ref5 == 127) | (LWIC_words.Ref6 == 127)
                        | (LWIC_words.Ref7 == 127) | (LWIC_words.Ref8 == 127)]
clean_negemo_word = negemo_word['Word'].str.replace('*', '')


anx_word = LWIC_words[(LWIC_words.Ref1 == 128) | (LWIC_words.Ref2 == 128) | (LWIC_words.Ref3 == 128)
                        | (LWIC_words.Ref4 == 128) | (LWIC_words.Ref5 == 128) | (LWIC_words.Ref6 == 128)
                        | (LWIC_words.Ref7 == 128) | (LWIC_words.Ref8 == 128)]

clean_anx_word = anx_word['Word'].str.replace('*', '')


health_word = LWIC_words[(LWIC_words.Ref1 == 148) | (LWIC_words.Ref2 == 148) | (LWIC_words.Ref3 == 148)
                        | (LWIC_words.Ref4 == 148) | (LWIC_words.Ref5 == 148) | (LWIC_words.Ref6 == 148)
                        | (LWIC_words.Ref7 == 148) | (LWIC_words.Ref8 == 148)]
clean_health_word = health_word['Word'].str.replace('*', '')


relig_word = LWIC_words[(LWIC_words.Ref1 == 359) | (LWIC_words.Ref2 == 359) | (LWIC_words.Ref3 == 359)
                        | (LWIC_words.Ref4 == 359) | (LWIC_words.Ref5 == 359) | (LWIC_words.Ref6 == 359)
                        | (LWIC_words.Ref7 == 359) | (LWIC_words.Ref8 == 359)]
clean_relig_word = relig_word['Word'].str.replace('*', '')

death_word = LWIC_words[(LWIC_words.Ref1 == 360) | (LWIC_words.Ref2 == 360) | (LWIC_words.Ref3 == 360)
                        | (LWIC_words.Ref4 == 360) | (LWIC_words.Ref5 == 360) | (LWIC_words.Ref6 == 360)
                        | (LWIC_words.Ref7 == 360) | (LWIC_words.Ref8 == 360)]
clean_death_word = death_word['Word'].str.replace('*', '')

swear_word_list = [word for word in swear_word['Word']]
family_word_list = [word for word in family_word['Word']]
negemo_word_list = [word for word in negemo_word['Word']]
anx_word_list = [word for word in anx_word['Word']]
health_word_list = [word for word in health_word['Word']]
relig_word_list = [word for word in relig_word['Word']]
death_word_list = [word for word in death_word['Word']]

clean_swear_word_list = [word for word in clean_swear_word]
clean_family_word_list = [word for word in clean_family_word]
clean_negemo_word_list = [word for word in clean_negemo_word]
clean_anx_word_list = [word for word in clean_anx_word]
clean_health_word_list = [word for word in clean_health_word]
clean_relig_word_list = [word for word in clean_relig_word]
clean_death_word_list = [word for word in clean_death_word]


LWIC_Categories = clean_swear_word_list + clean_family_word_list + clean_negemo_word_list + clean_anx_word_list +clean_health_word_list + clean_relig_word_list + clean_death_word_list


LWIC_Categories_unique_list = np.unique(LWIC_Categories).tolist()
print(len(LWIC_Categories_unique_list))

# tokenizer.add_tokens(LWIC_Categories_unique_list)
# print(len(tokenizer.get_vocab()))