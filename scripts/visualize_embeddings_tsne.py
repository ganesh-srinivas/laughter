"""Produce t-SNE plots for every embedding in given dataset"""

from sklearn.manifold import TSNE

import cPickle
import random

# File locations
AUDIO_EMBEDDINGS_DICT = "/home/gxs393/data/audio_embeddings_dict_9806_5klaughter_5knota.pkl"
AUDIO_LABEL_INDICES_DICT = "/home/gxs393/data/train_id2indlist.pkl"
OUTPUT_FILENAME = "tsne_output.csv"

EXAMPLES_SIZE_LIMIT = 1000
RANDOM_STATE = 0 # for reproducability

# Hyperparameters
# For information on how to tune perplexity, learning rate, etc. 
# see https://distill.pub/2016/misread-tsne/
N_COMPONENTS = 2
PERPLEXITY = 30
LEARNING_RATE = 200.0
N_ITER = 1000

def main():
    audio_embeddings_dict = cPickle.load(open(AUDIO_EMBEDDINGS_DICT, 'rb'))
    audio_label_indices_dict = cPickle.load(open(AUDIO_LABEL_INDICES_DICT, 'rb'))
    
    X = []
    ids = []
    for k in audio_embeddings_dict.keys()[:EXAMPLES_SIZE_LIMIT]:
       for embedding in audio_embeddings_dict[k]:
           X.append(embedding) 
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
