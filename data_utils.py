import os
import wget
import tarfile
import re
from nltk.tokenize import word_tokenize
import collections
import pandas as pd
import pickle
import numpy as np
import gensim
from gensim.models import Word2Vec
import multiprocessing
import csv
import warnings
 
from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors


warnings.filterwarnings("ignore", category=DeprecationWarning)

TRAIN_PATH = "yelp_review_polarity_csv/train.csv"
TEST_PATH = "yelp_review_polarity_csv/test.csv"


# def download_dbpedia():
#     dbpedia_url = 'https://github.com/le-scientifique/torchDatasets/raw/master/dbpedia_csv.tar.gz'

#     wget.download(dbpedia_url)
#     with tarfile.open("dbpedia_csv.tar.gz", "r:gz") as tar:
#         tar.extractall()


# def clean_str(text):
#     text = re.sub(r"[^A-Za-z0-9(),!?\'\`\"]", " ", text)
#     text = re.sub(r"\s{2,}", " ", text)
#     text = text.strip().lower()

#     return text


def build_word_dict():
    if not os.path.exists("word_dict.pickle"):
            
        print('Creating the new file')
        
        train_df = pd.read_csv(TRAIN_PATH, names=["class", "title", "content"])
        contents = train_df["content"]

        words = list()
        for content in contents:
            for word in word_tokenize(clean_str(content)):
                words.append(word)

        word_counter = collections.Counter(words).most_common()
        word_dict = dict()
        word_dict["<pad>"] = 0
        word_dict["<unk>"] = 1
        word_dict["<eos>"] = 2
        for word, _ in word_counter:
            word_dict[word] = len(word_dict)

        with open("word_dict.pickle", "wb") as f:
            pickle.dump(word_dict, f)

    else:
        with open("word_dict.pickle", "rb") as f:
            word_dict = pickle.load(f)

    return word_dict


def build_word_dataset(step, word_dict, document_max_len):
    if step == "train":
        print('hi')
        df = pd.read_csv(TRAIN_PATH, names=["class", "content"])
    else:
        df = pd.read_csv(TEST_PATH, names=["class", "content"])

    # Shuffle dataframe
    df = df.sample(frac=1)
    x = list(map(lambda d: word_tokenize(clean_str(d)), df["content"]))
    x = list(map(lambda d: list(map(lambda w: word_dict.get(w, word_dict["<unk>"]), d)), x))
    x = list(map(lambda d: d + [word_dict["<eos>"]], x))
    x = list(map(lambda d: d[:document_max_len], x))
    x = list(map(lambda d: d + (document_max_len - len(d)) * [word_dict["<pad>"]], x))

    y = list(map(lambda d: d - 1, list(df["class"])))
    
    

    return x, y

def tokenize_char(l):
    final = []
    for i in l:
        for j in i:
            val = [*j]
            for k in val:
                final.append(k.lower())
                
    return final


'''
def train_model(df):
    sentences = []
    count = 0
    for i in df['content']:
        count += 1
        
        sentences = sentences + [i.split()]
        
        if count % 5 == 0:
            break
            
    f_sentences = []
    for i in sentences:
        f_sentences.append(tokenize_char(i))
        
        
    print('Training skipgram model now....')
    w2v = Word2Vec(f_sentences, window = 5, min_count = 5, negative = 15, vector_size = 6, workers = multiprocessing.cpu_count())
    print('Training done!')
    print(w2v.wv.vectors.shape)
    print(w2v.wv['a'])
    print(w2v.wv.most_similar('i'))
    
    print('Saving model..')
    w2v.wv.save_word2vec_format('model_vectors/vectors.kv')
    print('Model saved successfully!')
'''
    


def build_char_dataset(step, model, document_max_len): #use the trained model here
    train_model_flag = False
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:’'\"/|_#$%ˆ&*˜‘+=<>()[]{} "
    if step == "train":


        df = pd.read_csv(TRAIN_PATH, names=["class","content"])
        
        train_model_flag = False
        # df = df.head(2000)
        
        # print(df.shape)
        # quit()
        
        

    else:
        df = pd.read_csv(TEST_PATH, names=["class", "content"])
        train_model_flag = False

    # Shuffle dataframe
    df = df.sample(frac=1)
    
    # print(df.head(2))

    char_dict = dict()
    rev_char_dict = dict()
    char_dict["<pad>"] = 0
    char_dict["<unk>"] = 1
    rev_char_dict[0] = "<pad>"
    rev_char_dict[1] = "<unk>"
    
    for c in alphabet:
        char_dict[c] = len(char_dict)
        rev_char_dict[len(char_dict) - 1] = c
        
        
    # print(char_dict)
    # quit()

        
    print('Training skip-gram model now...')
    if train_model_flag:
        train_model(df)
    
    print('Model trained and saved...')

    alphabet_size = len(alphabet) + 2

    x = list(map(lambda content: list(map(lambda d: char_dict.get(d, char_dict["<unk>"]), content.lower())), df["content"]))
    x = list(map(lambda d: d[:document_max_len], x))
    x = list(map(lambda d: d + (document_max_len - len(d)) * [char_dict["<pad>"]], x))

    y = list(map(lambda d: d - 1, list(df["class"])))
    
    
    return x, y, alphabet_size, rev_char_dict


def batch_iter(inputs, outputs, batch_size, num_epochs):
    inputs = np.array(inputs)
    outputs = np.array(outputs)

    num_batches_per_epoch = (len(inputs) - 1) // batch_size + 1
    for epoch in range(num_epochs):
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len(inputs))
            yield inputs[start_index:end_index], outputs[start_index:end_index]
