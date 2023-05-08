import argparse
import tensorflow as tf
from data_utils import *
from sklearn.model_selection import train_test_split
from cnn_models.word_cnn import WordCNN
from cnn_models.char_cnn import CharCNN
from cnn_models.vd_cnn import VDCNN
# from rnn_models.word_rnn import WordRNN
# from rnn_models.attention_rnn import AttentionRNN
# from rnn_models.rcnn import RCNN
import numpy as np
import gensim
from gensim.models import Word2Vec
tf.compat.v1.disable_eager_execution()

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# config = tf.compat.v1.ConfigProto()
# config.gpu_options.force_gpu_compatible = True
# with tf.compat.v1.Session(config = config) as sess:


import keras
config = tf.compat.v1.ConfigProto( device_count = {'GPU': 4 , 'CPU': 1} )
sess = tf.compat.v1.Session(config=config) 
keras.backend.set_session(sess)


TRAIN_PATH = "yelp_review_polarity_csv/train.csv"




def train_model(df):
    sentences = []
    count = 0
    for i in df['content']:
        # count += 1
        
        sentences = sentences + [i.split()]
        
        # if count % 5 == 0:
        #     break
            
    f_sentences = []
    for i in sentences:
        f_sentences.append(tokenize_char(i))
        
        
        
    print('Training skipgram model now....')
    w2v = Word2Vec(f_sentences, window = 5, min_count = 5, negative = 15, vector_size = 6)
    print('Training done!')
    print(w2v.wv.vectors.shape)
    print(w2v.wv['a'])
    print(w2v.wv.most_similar('i'))
    
    print('Saving model..')
    w2v.wv.save_word2vec_format('model_vectors/vectors.kv')
    print('Model saved successfully!')
    
def clean_str(text):
    text = re.sub(r"[^A-Za-z0-9(),!?\'\`\"]", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = text.strip().lower()

    return text


    
alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:’'\"/|_#$%ˆ&*˜‘+=<>()[]{} "

df = pd.read_csv(TRAIN_PATH, names=["class","content"])

df = df.sample(frac=1)

char_dict = dict()
rev_char_dict = dict()
char_dict["<pad>"] = 0
char_dict["<unk>"] = 1
rev_char_dict[0] = "<pad>"
rev_char_dict[1] = "<unk>"

for c in alphabet:
    char_dict[c] = len(char_dict)
    rev_char_dict[len(char_dict) - 1] = c

print('Training skipgram model')
print(df.head(10))
print(rev_char_dict)
# quit()
train_model(df)