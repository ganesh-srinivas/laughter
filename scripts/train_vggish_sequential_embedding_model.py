"""Train AudioSet bal_train embedding sequences using a bidirectional LSTM model"""

from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Bidirectional, LSTM, BatchNormalization, Dropout

from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np

# Standard libraries
import glob
import cPickle

# Filenames, Constants and hyperparameters
CLASS_LABELS_INDICES_FILENAME = "file_lists/class_labels_indices.csv"
LSTM_MODEL_FILENAME = "saved_models/bidirectional_lstm.h5"
TFRECORD_LOCATION = "/home/gxs393/audioset_v1_embeddings_incomplete/bal_train/"

N_CLASSES_ = 527


def get_multihot_encoding(x, class_ids=list(range(N_CLASSES_))):
    enc = []
    for i in class_ids:
        if i in x:
            enc.append(1)
        else:
            enc.append(0)
    return enc

def get_class_display_names(output_vector, classes2displaynames=None, n_class_indices_to_return=5):
    if classes2displaynames is None:
        with open(CLASS_LABELS_INDICES_FILENAME, "r") as fh:
            allclasses = fh.read().splitlines()
        classes2displaynames={int(i.split(',')[0]):i.split(',')[2] for i in allclasses[1:]}
    
    # Sort indices according to size of their values and 
    # then reverse the result to obtain class indices in descending order of confidence
    return np.argsort(output_vector)[::-1][:n_class_indices_to_return]


def main():
    with open(CLASS_LABELS_INDICES_FILENAME, "r") as fh:
        allclasses = fh.read().splitlines()
    classes2displaynames={int(i.split(',')[0]):i.split(',')[2] for i in allclasses[1:]}
    

    audio_embeddings_dict = {}
    audio_labels_dict = {}
    audio_multihot_dict = {}
    all_tfrecord_filenames = glob.glob(TFRECORD_LOCATION + "*.tfrecord")
    top_tfrecord_filenames = ['0W', '1_', '-g', '2V', '9Z', 'mh', '__', 'rT', '02', '07', '1e', '60', '-D', '-z', '2C', '2I', 'JL', 'Tg', 'Aa', '49', '4T', '9I', '0Q', 'CR', 'mY', 'er', '1P', '1j', 'W_', '6e', '-R', '-o', 'fw', '20', 'Ea', 'Ep', 'XY', '7I', 'pZ', 'OL', 'bj', 'Au', '3P', 'qp', 'P_', 'cQ', 'cT', 'Bj', 'GF', 'Gt', 'Z2', 'Zb', 'mO', 'mm', 'Lv', 'LM', '_0', 'd-', '0N', '0c', '0a', '0e', 'C9', 'CJ', '5J', '5D', 'H_', 'nj', 'nv', '1O', 'Wd', 'jt', 'ji', 'jO', '6o', '6z', '6X', 'IP', 'II', 'oG', 'oh', 'N3', '-J', '-f', '-m', '-7', '-6', 'SA', 'yB', 'E_', '7o', 'O7', 'AA', '33', 'Fd', 'Yj', 'Yx', '8o', 'K6', 'Pu', 'cA', 'wM', 'BO', 'BY', 'Be', 'Bc', 'Ui', 'UT', 'zY', '4v', '4l', 'G1', 'GW', 'ZN', 'ZC', 'Zo', '9C', 'LL', 'LJ', 'LK', 'LT', '_d', 'rN', 'Qt', 'd6', 'dl', '0S', '0X', '0q', 'CZ', 'CY', 'Ca', 'Cp', 'V9', '5T', '5B', '5f', '5a', '58', '55', 'HY', 'Hd', 'nF', 'M9', 'Mo', 'Mr', 'ML', '1J', '1y', '1a', '15', 'WQ', 'Wm', 'W4', '6i', '6C', '6F', 'yh', 'IM', 'IC', 'NX', 'aa', 'a9', '-U', '-u', 'S-', 'fz', '2R', '2X', 'yf', '2z', 'ET', 'Ef', '7v', '7K', 'kj', 'JW', 'pS', 'pD', 'pL', 'pj', 'p1', 'OM', 'OJ', 'b2', 'bu', 'u3', 'uC', 'AC', 'AE', 'AM', 'AO', 'Am', 'Ar', 'Aw', '3S', '3z', '39', 'FS', 'FF', 'FK', 'YT', 'Yn', 'lS', '8n', 'zQ', 'qF', 'qM', 'qy', 'PP', 'Uj', 'UP', 'hR', '4W', '4M', '4f', 'G6', 'G8', 'GS', 'GI', 'Gc', 'Z0', 'ZM', 'ZO', 'ZD', 'ZF', 'Zv', '9x', '9K', 'm2', '91', 'ml', 'mw', 'my', 'Lf', 'Lk', 'L9', '_X', '_H', '_k', 'ra', 'r7', 'QX', '06', '0H', '0n', 'C6', 'C4', 'CM', 'CC', 'CE', 'CU', 'Cn', 'Cz', 'Cr', 'it', 'VC', 'VK', '5P', '5F', '5q', '5h', '5g', '5b', '56', '5-', 'HZ', 'Hl', 'm8', 'nG', 'nr', 'mT', 'M1', 'ME', 'MO', 'sG', 'eg', 'ea', 'ec', 'eH', 'eX', '1T', '1G', '1d', '11', 'DX', 'DY', 'Dj', 'D-', 'WA', 'We', 'jl', 'jo', 'ja', 'jL', '68', '6g', '6k', '6m', '6U', 'IO', 'ID', 'Ih', 'TO', 'yw', 'oy', 'ou', 'Nl', 'Nf', 'NP', 'ae', 'aS', 'a2', '-G', '-F', '-H', '-X', '-v', 'iT', '-8', 'SH', 'fW', 'y5', '26', '2Z', '2M', '2r', '2y', '2g', '2a', '2c', '2h', '2k', 'Et', 'zd', 'Xo', '7L', '7S', '72', 'kk', 'pB', 'Ou', 'Oq', 'OW', 'bc', 'ua', 'Tl', 'Ti', 'TW', 'A8', 'AB', 'AT', 'AV', 'At', 'gu', 'gr', 'gX', 'g9', '3Z', '3O', 'zw', '3e', 'FT', 'FY', 'FM', 'YB', 'YV', 'Yy', 'lc', 'tN', '89', '84', '8m', '8p', '8N', '8L', '8X', '8T', 'Kz', 'KO', 'qW', 'qw', 'qh', 'wL', 'wE', 'zP', 'vf', 'vm', 'vZ', 'wR', 'BE', 'BQ', 'BZ', 'Bt', 'Uy', 'Ux', 'UM', 'UF', 'h9', 'h5', 'hZ', '4J', '4E', '4z', '4b', 'G_', 'GX', 'Ga', 'Gi', 'ZE', '9k', '9o', '9n', '9J', '9M', '9P', '9U', 'mg', 'mk', 'mt', 'LU', '_V', '_R', '_M', '_y', '_u', '_5', 'rJ', 'xw', 'rl', 'rn', 'Q9', 'Q-', 'Qn', 'QQ', 'QS', 'QD', 'QH', 'do', 'xb', 'dz', 'dP', '01', '0Y', '0A', '0E', '0L', '0f', 'C0', 'CN', 'CF', 'CS', 'CT', 'Ci', 'Ce', 'Cw', 'Cu', 'in', 'V6', 'VG', 'Vz', 'wc', '5Q', '5C', '5p', '5r', '5m', '5o', '5i', '5k', '59', 'HX', 'HR', 'HS', 'HH', 'HO', 'Hp', 'Ht', 'Hf', 'm-', 'm3', 'nJ', 'mU', 'n0', 'Mx', 'MI', 'tS', 's7', 'sY', 'sp', 'sv', 'su', 'sk', 'Rb', 'RQ']
    top_tfrecord_filenames = [TFRECORD_LOCATION+i+'.tfrecord' for i in top_tfrecord_filenames]
    # Load embeddings
    sess = tf.Session() 
    for tfrecord in top_tfrecord_filenames:
      for example in tf.python_io.tf_record_iterator(tfrecord):
        if len(audio_embeddings_dict) % 200 == 0:
          print "Saving dictionary: {}".format(len(audio_embeddings_dict))
          cPickle.dump(audio_embeddings_dict, open('audio_embeddings_dict_bal_train_{}.pkl'.format(len(audio_embeddings_dict)), 'wb'))
          cPickle.dump(audio_multihot_dict, open('audio_multihot_dict_bal_train_{}.pkl'.format(len(audio_multihot_dict)), 'wb'))
        tf_example = tf.train.Example.FromString(example)
        vid_id = tf_example.features.feature['video_id'].bytes_list.value[0].decode(encoding='UTF-8')

        example_label = list(np.asarray(tf_example.features.feature['labels'].int64_list.value))
        tf_seq_example = tf.train.SequenceExample.FromString(example)
        n_frames = len(tf_seq_example.feature_lists.feature_list['audio_embedding'].feature)
        audio_frame = []    
        for i in range(n_frames):
            audio_frame.append(tf.cast(tf.decode_raw(
                 tf_seq_example.feature_lists.feature_list['audio_embedding'].feature[i].bytes_list.value[0],tf.uint8)
                ,tf.float32).eval(session=sess))        
        audio_embeddings_dict[vid_id] = audio_frame
        audio_labels_dict[vid_id] = example_label
        audio_multihot_dict[vid_id] = get_multihot_encoding(example_label)
      if len(audio_embeddings_dict) % 200 == 0:
          print "Saving dictionary: {}".format(len(audio_embeddings_dict))
          cPickle.dump(audio_embeddings_dict, open('audio_embeddings_dict_bal_train_{}.pkl'.format(len(audio_embeddings_dict)), 'wb'))
          cPickle.dump(audio_multihot_dict, open('audio_multihot_dict_bal_train_{}.pkl'.format(len(audio_multihot_dict)), 'wb'))


    # Train-test split
    train, test = train_test_split(list(audio_labels_dict.keys()))
    xtrain = [audio_embeddings_dict[k] for k in train]
    ytrain = [audio_multihot_dict[k] for k in train]

    xtest = [audio_embeddings_dict[k] for k in test]
    ytest = [audio_multihot_dict[k] for k in test]

    # Pad all inputs to have constant sequence length
    maxlen = 10
    x_train = pad_sequences(xtrain, maxlen=maxlen)
    x_test = pad_sequences(xtest, maxlen=maxlen)

    y_train = np.asarray(ytrain)
    y_test = np.asarray(ytest)

    # Define sequential model in Keras
    print('Building model...')
    
    model = Sequential()
    model.add(BatchNormalization(input_shape=(maxlen, 128)))
    model.add(Dropout(.5))
    model.add(Bidirectional(LSTM(128, init='normal', activation='relu')))
    model.add(Dense(N_CLASSES_, activation='sigmoid', init='normal'))

    # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # Train sequential model
    print('Train...')
    model.fit(x_train, y_train,
              batch_size=BATCH_SIZE_,
              nb_epoch=NB_EPOCH,validation_data=(x_test, y_test))
    model.save(LSTM_MODEL_FILENAME, overwrite=True)
    
    # Get test set accuracy
    score, acc = model.evaluate(x_test, y_test, batch_size=64)
    print('Test score:', score)
    print('Test accuracy:', acc)

if __name__ == "__main__":
    main()

