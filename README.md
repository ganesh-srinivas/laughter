# Learning embeddings for laughter categorization

## Ganesh Srinivas

## Mentors: mpac and Vera Tobin

## [Proposal abstract](https://summerofcode.withgoogle.com/projects/#5795091189858304)
I propose to train a deep neural network to discriminate between various kinds of laughter (giggle, snicker, etc.) A convolutional neural network can be trained to produce continuous-valued vector representations (embeddings) for spectrograms of audio data. A triplet-loss function during training can constrain the network to learn an embedding space where Euclidean distance corresponds to acoustic similarity. In such a space, algorithms like k-Nearest Neighbors can be used for classification. The network weights can be visualized to glean insight about the low- and high-level features it has learned to look for (pitch, timbre, unknowns, etc.) I also propose to obtain visualizations of the embedding space of laughter sounds using dimension reduction techniques like Principal Components Analysis (PCA) and t-distributed Stochastic Neighbor Embedding (t-SNE). I will also apply these same techniques techniques directly on the high-dimension audio spectrograms. All techniques proposed here have been applied previously on related problems in audio and image processing.

Detailed proposal is also [available](redhen2017_proposal_ganesh_srinivas.pdf).

## Laughter Categorization Models available for usage
1. A TensorFlow + Keras implementation of a laughter categorization network: Google's VGGish model (TF) will convert every second of the input clip to a 128-dimension embedding, and a Bidirectional LSTM model (Keras) will produce labels from the sequence of embeddings. Top-1 label accuracy of 67% for categorization and 90% accuracy for detection (laughter-or-not).

`python scripts/predict_vggish_sequential_embedding_model.py`

2. A pure TensorFlow implementation of a laughter categorization network: a convolutional network that produces that tells whether the input belongs to one of six classes (baby laughter, belly laughter, chuckle/chortle, snicker, giggle, none of the above).

`python scripts/predict_convnet_10sec_model.py`

Code for a few more laughter categorization models that performed worse than these two is included. The saved models corresponding to those may be included if Github allows upload of these files (space constraint).

## Laughter visualization using t-SNE
1. A script that transforms audio clips (from a dataset of laughter and non-laughter examples) into points in 2D space i.e., produce a map of various sounds.

`python scripts/visualize_embeddings_tsne.py`



## Requirements
- TensorFlow
- keras
- librosa
- sklearn
- scipy
- numpy
- audioread


