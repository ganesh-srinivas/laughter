from keras.models import load_model
from keras.metrics import top_k_categorical_accuracy
from keras.preprocessing.sequence import pad_sequences

import numpy as np
import librosa
import six
import tensorflow as tf

import glob

import support_vggish_input
import support_vggish_params
import support_vggish_postprocess
import support_vggish_slim

PREDICTION_SEQUENCE_LENGTH = 5 # a prediction is made for every segment of this duration 

AUDIO_CLIPS_FOLDER = "/home/gxs393/dataset/redhendata/"
AUDIO_CLIPS_EXTENSION = "wav"

PCA_PARAMS = "saved_models/vggish_pca_params.npz"
CHECKPOINT = "saved_models/vggish_model.ckpt"
KERAS_SEQUENTIAL_MODEL = "saved_models/67bidirectional_lstm_dropout_batchnorm_sgd.h5"

MAX_SEQUENCE_LENGTH = 10 # should be same as keras model's max sequence length

def top_one_accuracy(x, y):
    """Must be passed to `load_model` function in `keras` as a custom_object"""
    return top_k_categorical_accuracy(x, y, k=1)

def get_top_class_display_name(output_vector):        
    """Returns the name of the class with the highest activation"""
    labels = ["laughter", "baby laughter", "giggle", "snicker", \
    "belly laugh", "chuckle/chortle", "none of the above"]
    sorted_indices = list(np.argsort(output_vector)[::-1])
    return labels[sorted_indices[0]]

def main(_):
    wav_files = glob.glob("../*.wav") 
    #wav_files = glob.glob(AUDIO_CLIPS_FOLDER + '*' + AUDIO_CLIPS_EXTENSION)
    examples_batch = {wav_file:support_vggish_input.wavfile_to_examples(wav_file) for wav_file in wav_files}

    # Prepare a postprocessor to munge the model embeddings.
    pproc = support_vggish_postprocess.Postprocessor(PCA_PARAMS)

    with tf.Graph().as_default(), tf.Session() as sess:
        # Define the model in inference mode, load the checkpoint, and
        # locate input and output tensors.
        support_vggish_slim.define_vggish_slim(training=False)
        support_vggish_slim.load_vggish_slim_checkpoint(sess, CHECKPOINT)
        features_tensor = sess.graph.get_tensor_by_name(
            support_vggish_params.INPUT_TENSOR_NAME)
        embedding_tensor = sess.graph.get_tensor_by_name(
            support_vggish_params.OUTPUT_TENSOR_NAME)

        # Run inference and postprocessing.
        embedding_batch = {}
        postprocessed_batch = {}
        for k in examples_batch.keys():
            embedding_batch[k] = sess.run([embedding_tensor], 
                                 feed_dict={features_tensor: examples_batch[k]})
            postprocessed_batch[k] = pproc.postprocess(np.asarray(embedding_batch[k][0]))
    
    sequential_model = load_model(KERAS_SEQUENTIAL_MODEL, custom_objects={'<lambda>': top_one_accuracy})

    # Presenting the predictions over `x` second segments in the audio clip(s)
    for k in postprocessed_batch.keys():
        print "FILENAME: {}".format(k)
        for i in range(0, len(postprocessed_batch[k]), PREDICTION_SEQUENCE_LENGTH):
            # Presenting the predictions over `PREDICTION_SEQUENCE_LENGTH` second segments in the audio clip(s)
            print "start:", i, " stop:", i+PREDICTION_SEQUENCE_LENGTH,

            sequential_input = pad_sequences([postprocessed_batch[k][i: i + PREDICTION_SEQUENCE_LENGTH]], MAX_SEQUENCE_LENGTH)
            predictions = sequential_model.predict(sequential_input)

            print get_top_class_display_name(predictions[0])
        print "###"         

if __name__ == '__main__':
  tf.app.run()
