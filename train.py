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
import matplotlib
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
tf.compat.v1.disable_eager_execution()

import keras
config = tf.compat.v1.ConfigProto( device_count = {'GPU': 4 , 'CPU': 1} )
sess = tf.compat.v1.Session(config=config) 
keras.backend.set_session(sess)


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
# b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
# c = tf.matmul(a, b)

training_loss = []
training_loss_step = []
validation_loss = []
validation_loss_step = []

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="word_cnn",
                    help="word_cnn | char_cnn | vd_cnn | word_rnn | att_rnn | rcnn")
args = parser.parse_args()

# if not os.path.exists("dbpedia_csv"):
#     print("Downloading dbpedia dataset...")
#     download_dbpedia()

NUM_CLASS = 4
BATCH_SIZE = 64
NUM_EPOCHS = 10
WORD_MAX_LEN = 100
CHAR_MAX_LEN = 1014

char_dict = None
print("Building dataset...")
if args.model == "char_cnn":
    x, y, alphabet_size, char_dict = build_char_dataset("train", "char_cnn", CHAR_MAX_LEN)
elif args.model == "vd_cnn":
    x, y, alphabet_size, char_dict = build_char_dataset("train", "vdcnn", CHAR_MAX_LEN)

# print(char_dict)
# quit()
f_x = []
# print(x[:2])

#integer --> character
for i in x:
    f = []

    # print(i)
    for j in i:
        # print(j)
        # if j in char_dict.keys(): 
        #Get the character --> Use the keyed vector to get the distributed representation
        f.append(char_dict[j])

    # if count % 10 == 0:
    #     print(count)


    f_x.append(f)

#character --> Distributed Representation
from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format('model_vectors/vectors.kv')
f_f_x = []

# print(f_x[:2])

l = []
for i in f_x:
    f_f_x_1 = []
    for j in i:
        l1 = []
        if j in model.key_to_index:
            l = model[j]

        else:
            l = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        for k in l:
            v = np.array([k])
            v = v.astype(np.float32)
            l1.append(v)

        # f_arr = arr.astype(np.float32)
        f_f_x_1.append(l1)

    f_f_x_2 = np.array(f_f_x_1)
    f_f_x.append(f_f_x_2)

final_arr = f_f_x



print('Printing f_x...')

train_x, valid_x, train_y, valid_y = train_test_split(final_arr, y, test_size=0.15)

print('Starting model training')
with tf.compat.v1.Session() as sess:  
    if args.model == "word_cnn":
        model = WordCNN(vocabulary_size, WORD_MAX_LEN, NUM_CLASS)
    elif args.model == "char_cnn":
        model = CharCNN(alphabet_size, CHAR_MAX_LEN, NUM_CLASS)
    else:
        raise NotImplementedError()

    sess.run(tf.compat.v1.global_variables_initializer())
    saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())


    train_batches = batch_iter(train_x, train_y, BATCH_SIZE, NUM_EPOCHS)
    num_batches_per_epoch = (len(train_x) - 1) // BATCH_SIZE + 1
    max_accuracy = 0

    print('HI')


    for x_batch, y_batch in train_batches:
        train_feed_dict = {
            model.x: x_batch,
            model.y: y_batch,
            model.is_training: True
        }

        _, step, loss = sess.run([model.optimizer, model.global_step, model.loss], feed_dict=train_feed_dict)


        if step % 1000 == 0:
            print("step {0}: loss = {1}".format(step, loss))
            training_loss.append(loss)
            training_loss_step.append(step)

        if step % 2000 == 0:
            # Test accuracy with validation data for each epoch.
            valid_batches = batch_iter(valid_x, valid_y, BATCH_SIZE, 1)
            sum_accuracy, cnt = 0, 0

            for valid_x_batch, valid_y_batch in valid_batches:
                valid_feed_dict = {
                    model.x: valid_x_batch,
                    model.y: valid_y_batch,
                    model.is_training: False
                }

                # print(type(model.x))

                accuracy = sess.run(model.accuracy, feed_dict=valid_feed_dict)
                sum_accuracy += accuracy
                cnt += 1
            valid_accuracy = sum_accuracy / cnt
            validation_loss.append(valid_accuracy)
            validation_loss_step.append(step)

            print("\nValidation Accuracy = {1}\n".format(step // num_batches_per_epoch, sum_accuracy / cnt))

            # Save model
            print('Printing valid accuracy')
            print(valid_accuracy)
            
            

            if valid_accuracy > max_accuracy:
                max_accuracy = valid_accuracy
                saver.save(sess, "{0}/{1}.ckpt".format(args.model, args.model), global_step=step)
                print("Model is saved.\n")
                
print('Plotting')
plt.plot(training_loss_step,training_loss, '-ob')
plt.xlabel('Step')
plt.ylabel('Training Loss')
plt.show()
plt.savefig('training_accuracy.png')
plt.close()

plt.plot(validation_loss_step, validation_loss, '-or')
plt.xlabel('Step')
plt.ylabel('Validaton Accuracy')
plt.show()
plt.savefig('validation_accuracy.png')
plt.close()
                






