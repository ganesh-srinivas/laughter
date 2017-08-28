"""
This script contains code for a Convolutional Neural Network
that classifies a 10-sec audio clip as either containing laughter 
or not.

Author: Ganesh Srinivas <gs401 [at] snu.edu.in>
"""

import glob
import os
import random
import subprocess

import tensorflow as tf
import numpy as np
import librosa

#import support_feature_extraction

## Load trained models
IMPORT_META_GRAPH = "/home/gxs393/experiments/eleventh_experiment/model-19340.meta"
IMPORT_LATEST_CHECKPOINT = "/home/gxs393/experiments/eleventh_experiment/"

## Files to predict on 
FILENAMES = "../dataset/unbalanced/10secondclipfiles.txt"
DATASET_LOCATION = "../dataset/unbalanced/"

possible_categories = ['laughter','nota_clips']


## Hyperparameters
# for Learning algorithm
learning_rate = 0.0001
batch_size = 120 
training_iterations = 195000

# for Feature extraction
max_audio_length = 221184
frames = 433
bands = 120 
feature_size = frames*bands #433x60

# for Network
num_labels = 2
num_channels = 2 

kernel_size = 30
depth = 20
num_hidden = 200

num_hidden2 = 100
num_hidden3 = 50

## Helper functions for loading data and extracting features
def labeltext2labelid(category_name):
    """
    Returns a numerical label for each laughter category
    """
    possible_categories = ['laughter',
    'nota_clips']
    return possible_categories.index(category_name)

def shape_sound_clip(sound_clip, required_length=max_audio_length):
    """
    Shapes sound clips to have constant length
    """
    difference = required_length-sound_clip.shape[0]
    if difference == 0:
        return sound_clip
    elif difference < 0:
        # Clip length exceeds required length. Trim it.
        modified_sound_clip = sound_clip[:-difference]
        return modified_sound_clip
    else:
        z = np.zeros((required_length - sound_clip.shape[0],))
        modified_sound_clip = np.append(sound_clip, z)
        return modified_sound_clip

def extract_features_from_waveforms(waveforms):
    """
    Extract log-scaled mel-spectrograms and their corresponding 
    deltas from the audio waveform (not the filename)
    """
    log_specgrams = []
    #labels=[]
    for s in waveforms:
      sound_clip = shape_sound_clip(s)

      melspec = librosa.feature.melspectrogram(sound_clip, n_mels = 120, n_fft=1024)
      #print melspec.shape

      logspec = librosa.power_to_db(melspec, ref = np.max)
      #print logspec.shape
      logspec = logspec.T.flatten()[:, np.newaxis].T
      #print logspec.shape

      #print "Produce of two elements in melspec: ", melspec.shape[0]*melspec.shape[1]  
      log_specgrams.append(logspec)
      del sound_clip
      del melspec
      del logspec
      #labels.append(labeltext2labelid(f.split('/')[-2]))  

    log_specgrams=np.asarray(log_specgrams).reshape(len(log_specgrams),bands,frames,1)

    features = np.concatenate((log_specgrams, np.zeros(np.shape(log_specgrams))), axis=3)

    for i in range(len(features)):
          features[i, :, :, 1] = librosa.feature.delta(features[i, :, :, 0])

    return np.array(features)#, np.array(labels,dtype=np.int)


def extract_features(filenames):
    """
    Extract log-scaled mel-spectrograms and their corresponding 
    deltas from the sound clips
    """
    log_specgrams = []
    labels=[]
    for f in filenames:
      signal,s = librosa.load(f)
      sound_clip = shape_sound_clip(signal)

      melspec = librosa.feature.melspectrogram(sound_clip, n_mels = 120, n_fft=1024)
      #print melspec.shape

      logspec = librosa.power_to_db(melspec, ref = np.max)
      #print logspec.shape
      logspec = logspec.T.flatten()[:, np.newaxis].T
      #print logspec.shape

      #print "Produce of two elements in melspec: ", melspec.shape[0]*melspec.shape[1]  
      log_specgrams.append(logspec)
      del signal
      del sound_clip
      del melspec
      del logspec
      if f.split('/')[-2] == 'nota_clips':
          label = labeltext2labelid('nota_clips')
      else:
          label = labeltext2labelid('laughter')
      #print "label: ", label
      labels.append(label)

    log_specgrams=np.asarray(log_specgrams).reshape(len(log_specgrams),bands,frames,1)

    features = np.concatenate((log_specgrams, np.zeros(np.shape(log_specgrams))), axis=3)

    for i in range(len(features)):
          features[i, :, :, 1] = librosa.feature.delta(features[i, :, :, 0])

    return np.array(features), np.array(labels,dtype=np.int)

def one_hot_encode(labels, num_labels=num_labels):
    """
    Convert list of label IDs to a list of one-hot encoding vectors
    """
    n_labels = len(labels)
    n_unique_labels = num_labels

    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1

    return one_hot_encode

## Helper functions for defining the network
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(.1, shape = shape)
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


## Defining the network as a TensorFlow computational graph
X = tf.placeholder(tf.float32, shape=[None,bands,frames,num_channels])
Y = tf.placeholder(tf.float32, shape=[None,num_labels])

# normalization
X_normalized = tf.nn.l2_normalize(X, dim = 0)

cov = apply_convolution(X_normalized,kernel_size,num_channels,depth)
shape = cov.get_shape().as_list()
#cov2 = apply_convolution(cov, kernel_size, num_channels, depth)
#shape2 = cov2.get_shape().as_list()
cov_flat = tf.reshape(cov, [-1, shape[1] * shape[2] * shape[3]])

f_weights = weight_variable([shape[1] * shape[2] * depth, num_hidden])
f_biases = bias_variable([num_hidden])

f = tf.nn.sigmoid(tf.add(tf.matmul(cov_flat, f_weights),f_biases))

out_weights = weight_variable([num_hidden, num_hidden2])
out_biases = bias_variable([num_hidden2])

f2 = tf.matmul(f, out_weights) + out_biases
out_weights2 = weight_variable([num_hidden2, num_labels])
out_biases2 = bias_variable([num_labels])
pred = tf.nn.softmax(tf.matmul(f2, out_weights2) + out_biases2)

# Defining the loss function 
cross_entropy = -tf.reduce_sum(Y * tf.log(pred))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

#train_prediction = tf.nn.softmax(cross_entropy)
correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def predict_on_long_clips():
    """Load the saved model and perform inference/prediction on features obtained from inputs. 
    Splits the audio into 10second chunks and predicts on those chunks."""
    with open(FILENAMES,"r") as fh:
        filecontents=fh.read()
        filenames=filecontents.splitlines()
        random.shuffle(filenames)
        filenames=filenames[:5] #[:5] is for quickly verifying if things work
        filenames = [DATASET_LOCATION+f for f in filenames]

    session = tf.Session()
    saver = tf.train.import_meta_graph(IMPORT_META_GRAPH)
    saver.restore(session, tf.train.latest_checkpoint(IMPORT_LATEST_CHECKPOINT))
    tf.global_variables_initializer().run(session=session)

    test_x = {}
    for f in filenames:
        s, sr = librosa.load(f)
        total_chunks = s.shape[0]/max_audio_length
        waveforms = [s[max_audio_length*i:max_audio_length*(i+1)] for i in range(total_chunks)]
        test_x[f] = extract_features_from_waveforms(waveforms)

        print "FILENAME: ", f
        predictions = session.run(tf.argmax(pred, 1), feed_dict={X: test_x[f]})
        print [possible_categories[p] for p in predictions]


predict_on_long_clips()
