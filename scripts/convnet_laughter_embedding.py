import glob
import os
import random

import librosa
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

plt.ion()

FILENAMES = "../dataset/audioset_laughter_clips/10secondclipfiles.txt"
DATASET_LOCATION = "../dataset/audioset_laughter_clips/"
with open(FILENAMES,"r") as fh:
    filecontents=fh.read()
    filenames=filecontents.split('\n')
    filenames=filenames[:-1]
    filenames = [DATASET_LOCATION+f for f in filenames]

random.shuffle(filenames)
filenames = filenames[:2250]
rnd_indices = np.random.rand(len(filenames)) < 0.70
print len(rnd_indices)
train = []
test = []
for i in range(len(filenames)):
    if rnd_indices[i]:
        train.append(rnd_indices)
    else:
        test.append(rnd_indices)
#train = filenames[rnd_indices]
#test = filenames[~rnd_indices]
print "Train: ", len(train)
print "Test: ", len(test)


def labeltext2labelid(text):
    possible_labels = ['baby_laughter', 'belly_laugh', 'chuckle_chortle', 'giggle', 'snicker']
    return possible_labels.index(text)

def shape_sound_clip(sound_clip, required_length=221184):
    z=np.zeros((required_length-sound_clip.shape[0],))
    return np.append(sound_clip,z)

def extract_features(filenames):
  log_specgrams = []
  labels=[]
  for f in filenames:
    signal,s = librosa.load(f)
    sound_clip = shape_sound_clip(signal)
    melspec = librosa.feature.melspectrogram(sound_clip, n_mels = 60)
    #print melspec.shape
    logspec = librosa.logamplitude(melspec)
    #print logspec.shape
    logspec = logspec.T.flatten()[:, np.newaxis].T
    #print logspec.shape
    #print "Produce of two elements in melspec: ", melspec.shape[0]*melspec.shape[1]  
    log_specgrams.append(logspec)
    labels.append(labeltext2labelid(f.split('/')[-2]))
    
  log_specgrams=np.asarray(log_specgrams).reshape(len(log_specgrams),60,433,1)
  features = np.concatenate((log_specgrams, np.zeros(np.shape(log_specgrams))), axis=3)
  for i in range(len(features)):
        features[i, :, :, 1] = librosa.feature.delta(features[i, :, :, 0])
  return np.array(features), np.array(labels,dtype=np.int)

def one_hot_encode(labels):
    n_labels = len(labels)
    #n_unique_labels = len(np.unique(labels))
    n_unique_labels = 5
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(1.0, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x,W,strides=[1,2,2,1], padding='SAME')

def apply_convolution(x,kernel_size,num_channels,depth):
    weights = weight_variable([kernel_size, kernel_size, num_channels, depth])
    biases = bias_variable([depth])
    return tf.nn.relu(tf.add(conv2d(x, weights),biases))

def apply_max_pool(x,kernel_size,stride_size):
    return tf.nn.max_pool(x, ksize=[1, kernel_size, kernel_size, 1], 
                          strides=[1, stride_size, stride_size, 1], padding='SAME')


frames = 433
bands = 60

feature_size = frames*bands #433x60
num_labels = 5
num_channels = 2

kernel_size = 30
depth = 20
num_hidden = 200

learning_rate = 0.01

X = tf.placeholder(tf.float32, shape=[None,bands,frames,num_channels])
Y = tf.placeholder(tf.float32, shape=[None,num_labels])
cov = apply_convolution(X,kernel_size,num_channels,depth)
shape = cov.get_shape().as_list()
cov_flat = tf.reshape(cov, [-1, shape[1] * shape[2] * shape[3]])
f_weights = weight_variable([shape[1] * shape[2] * depth, num_hidden])
f_biases = bias_variable([num_hidden])
f = tf.nn.sigmoid(tf.add(tf.matmul(cov_flat, f_weights),f_biases))
out_weights = weight_variable([num_hidden, num_labels])
out_biases = bias_variable([num_labels])
y_ = tf.nn.softmax(tf.matmul(f, out_weights) + out_biases)

cross_entropy = -tf.reduce_sum(Y * tf.log(y_))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
#train_prediction = tf.nn.softmax(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

batch_size = 225
training_iterations = 10
cost_history = np.empty(shape=[1],dtype=float)
with tf.Session() as session:
    tf.initialize_all_variables().run()
    for itr in range(training_iterations):    
        offset = (itr * batch_size) % (len(train) - batch_size)
        print offset
        batch = filenames[offset:(offset + batch_size)]
        batch_x, batch_y = extract_features(batch)
        batch_y = one_hot_encode(batch_y)
        print batch_y.shape, batch_x.shape
        _, c = session.run([optimizer, cross_entropy],feed_dict={X: batch_x, Y : batch_y})
        cost_history = np.append(cost_history,c)
    test_x, test_y = extract_features(test)
    print('Test accuracy: ',round(session.run(accuracy, feed_dict={X: test_x, Y: test_y}) , 3))
    fig = plt.figure(figsize=(15,10))
    plt.plot(cost_history)
    plt.axis([0,training_iterations,0,np.max(cost_history)])
    plt.show()

