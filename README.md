# laughter
Learning embeddings for laughter categorization
Ganesh Srinivas
Mentors: mpac, Vera Tobin and Otto Santa Ana

## (Proposal abstract)[https://summerofcode.withgoogle.com/projects/#5795091189858304]
I propose to train a deep neural network to discriminate between various kinds of laughter (giggle, snicker, etc.) A convolutional neural network can be trained to produce continuous-valued vector representations (embeddings) for spectrograms of audio data. A triplet-loss function during training can constrain the network to learn an embedding space where Euclidean distance corresponds to acoustic similarity. In such a space, algorithms like k-Nearest Neighbors can be used for classification. The network weights can be visualized to glean insight about the low- and high-level features it has learned to look for (pitch, timbre, unknowns, etc.) I also propose to obtain visualizations of the embedding space of laughter sounds using dimension reduction techniques like Principal Components Analysis (PCA) and t-distributed Stochastic Neighbor Embedding (t-SNE). I will also apply these same techniques techniques directly on the high-dimension audio spectrograms. All techniques proposed here have been applied previously on related problems in audio and image processing.

Detailed proposal is also (available)[redhen2017_proposal_ganesh_srinivas].

## Scripts
1. Tensorflow implementation of laughter embedding network: a feedforward attention-based convolutional network that produces 128-dimension embeddings using a triplet-loss function.
2. A script that takes 128-dimension audio embeddings as input and reduces dimensionality to 2D/3D using PCA/t-SNE using Scikit-Learn’s/van der Maaten’s implementation.

## Requirements
TensorFlow
sklearn

