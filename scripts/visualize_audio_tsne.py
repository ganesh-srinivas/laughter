"""Produce t-SNE plots for every audio clip in given dataset. We compute 
features from the spectrogram of the audio clip."""

import numpy as np
import librosa
from sklearn.manifold import TSNE

import cPickle
import random

# File locations
# AUDIO_FILENAMES_DICT should map video_ids to filenames.
AUDIO_FILENAMES_DICT = "/home/gxs393/data/audio_filenames_dict.pkl"
AUDIO_LABEL_INDICES_DICT = "/home/gxs393/data/train_id2indlist.pkl"
OUTPUT_FILENAME = "audio_tsne_output.csv"

MAX_AUDIO_LENGTH = 221184
EXAMPLES_SIZE_LIMIT = 1000
RANDOM_STATE = 0 # for reproducability

# Hyperparameters
# For information on how to tune perplexity, learning rate, etc. 
# see https://distill.pub/2016/misread-tsne/
N_COMPONENTS = 2
PERPLEXITY = 30
LEARNING_RATE = 200.0
N_ITER = 1000


def shape_sound_clip(sound_clip, required_length=MAX_AUDIO_LENGTH):
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

def extract_features(filename):
    y, sr = librosa.load(filename)
    y = shape_sound_clip(y)
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
    log_S = librosa.logamplitude(S, ref_power=np.max)
    mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)
    return mfcc.flatten()
    
def main():
    audio_label_indices_dict = cPickle.load(open(AUDIO_LABEL_INDICES_DICT, 'rb'))
    audio_filenames_dict = cPickle.load(open(AUDIO_FILENAMES_DICT, 'rb'))

    X = []
    ids = []

    for k in audio_filenames_dict.keys()[:EXAMPLES_SIZE_LIMIT]:
       X.append(extract_features(audio_filenames_dict[k]))
       ids.append(audio_label_indices_dict[k])

    # Apply t-SNE
    tsne = TSNE(n_components=N_COMPONENTS, perplexity=PERPLEXITY, \
                learning_rate=LEARNING_RATE, n_iter=N_ITER)
    Xtransformed = tsne.fit_transform(X)

    # save the embeddings along with the list of class IDs associated with
    # the clip from which it was taken.
    
    # Header for output file
    if N_COMPONENTS == 2:
        output_lines = ["dim1,dim2,labels"]
    elif N_COMPONENTS == 3:
        output_lines = ["dim1,dim2,dim3,labels"]

    for i in range(len(Xtransformed)):
        output_lines.append(",".join([str(j) for j in Xtransformed[i]])+ \
                            "," + ",".join([str(k) for k in ids[i]]))
    
    output_file_contents = "\n".join(output_lines) 
    with open(OUTPUT_FILENAME, 'w') as fh:
        fh.write(output_file_contents)
 
    
if __name__ == "__main__":
     main()
