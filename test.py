import tensorflow as tf
import argparse
from data_utils import *


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="word_cnn",
                    help="word_cnn | char_cnn | vd_cnn | word_rnn | att_rnn | rcnn")
args = parser.parse_args()

BATCH_SIZE = 128
WORD_MAX_LEN = 100
CHAR_MAX_LEN = 1014

if args.model == "char_cnn":
    test_x, test_y, alphabet_size, char_dict = build_char_dataset("test", "char_cnn", CHAR_MAX_LEN)
elif args.model == "vd_cnn":
    test_x, test_y, alphabet_size, char_dict = build_char_dataset("test", "vdcnn", CHAR_MAX_LEN)
else:
    word_dict = build_word_dict()
    test_x, test_y = build_word_dataset("test", word_dict, WORD_MAX_LEN)    

checkpoint_file = tf.train.latest_checkpoint(args.model)

f_x = []
# print(x[:2])

#integer --> character
print('Integer to Character')

# for i in test_x:
#     f = []

#     # print(i)
#     for j in i:
#         # print(j)
#         # if j in char_dict.keys(): 
#         #Get the character --> Use the keyed vector to get the distributed representation
#         f.append(char_dict[j])

#     # if count % 10 == 0:
#     #     print(count)


#     f_x.append(f)

#character --> Distributed Representation
from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format('model_vectors/vectors.kv')
f_f_x = []

# print(char_dict)
# quit()
print('Character to distributed representation')

# print(f_x[:2])

l = []
# print(test_x[:2])
# quit()

for i in test_x:
    f_f_x_1 = []
    # print(i)
    # quit()
    for j in i:
        l1 = []
        j1 = char_dict[j]
        if j1 in model.key_to_index:
            l = model[j1]
        # print('hi')
        # quit()
            
        else:
            l = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # print(l)
        # quit()
        for k in l:
            v = np.array([k])
            v = v.astype(np.float32)
            l1.append(v)
    
        # f_arr = arr.astype(np.float32)
        f_f_x_1.append(l1)
    # print(l)
    # quit()
        
    # print(f_f_x_1)
    # quit()
            
    f_f_x_2 = np.array(f_f_x_1)
    f_f_x.append(f_f_x_2)
    
final_arr = f_f_x


print(type(final_arr))
    
    
    
graph = tf.Graph()
with graph.as_default():
    with tf.compat.v1.Session() as sess:
        
        saver = tf.compat.v1.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        x = graph.get_operation_by_name("x").outputs[0]
        y = graph.get_operation_by_name("y").outputs[0]
        # a = sess.run(x)
        # print(x.shape)
        # quit()
        is_training = graph.get_operation_by_name("is_training").outputs[0]
        accuracy = graph.get_operation_by_name("accuracy/accuracy").outputs[0]
        # print(test_x[0])
        # quit()
        
        # print(f_x)
        # quit()
        
        batches = batch_iter(final_arr, test_y, BATCH_SIZE, 1)
        sum_accuracy, cnt = 0, 0
        for batch_x, batch_y in batches:
            feed_dict = {
                x: batch_x,
                y: batch_y,
                is_training: False
            }

            accuracy_out = sess.run(accuracy, feed_dict=feed_dict)
            sum_accuracy += accuracy_out
            cnt += 1

        print("Test Accuracy : {0}".format(sum_accuracy / cnt))
