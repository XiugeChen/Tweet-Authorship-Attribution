import numpy as np
import pandas as pd
import string
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Dense, Activation, Convolution1D, MaxPooling1D, Flatten, Dropout, Input, Embedding
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, LabelEncoder

import datetime

TRAIN_FILE = "../resources/data/train_tweets.txt"
SML_TRAIN_FILE = "../resources/data/test.txt"
TEST_FILE = "../resources/data/test_tweets_unlabeled.txt"

# Maximum length. Longer gets chopped. Shorter gets padded.
maxlen = 140

# Model params
# Filters for conv layers
nb_filter = 500
# Number of units in the dense layer
dense_outputs = 256
# Conv layer kernel size
filter_kernels = [3, 4, 5]
# Number of units in the final output layer. Number of classes.
cat_output = 236

# Compile/fit params
batch_size = 32
nb_epoch = 20


def create_vocab_set():
    alphabet = (list(string.ascii_lowercase) + list(string.digits) +
                list(string.punctuation) + ['\n'] + [' '])
    vocab_size = len(alphabet)
    check = set(alphabet)

    vocab = {}
    reverse_vocab = {}
    for ix, t in enumerate(alphabet):
        vocab[t] = ix
        reverse_vocab[ix] = t

    return vocab, reverse_vocab, vocab_size, check


def bigram_encode(x, maxlen, vocab, vocab_size, check):

    input_data = np.zeros((len(x), maxlen))
    for dix, sent in enumerate(x):
        counter = 0
        chars = list(sent.lower().replace(" ", ""))
        chars2 = []
        for i in range(len(chars)-1):
            chars2.append(chars[i] + chars[i+1])
        for i, c in enumerate(chars2):
            if counter >= maxlen:
                pass
            else:
                if c in check:
                    counter += 1
                    ix = vocab[c]
                    input_data[dix, counter-1] = ix

    return input_data


def mini_batch_generator(x, y, vocab, vocab_size, vocab_check, maxlen,
                         batch_size=128):

    for i in range(0, len(x), batch_size):
        x_sample = x[i:i + batch_size]
        y_sample = y[i:i + batch_size]

        input_data = bigram_encode(x_sample, maxlen, vocab, vocab_size,
                                   vocab_check)
        yield (input_data, y_sample)


def shuffle_matrix(x, y):
    # # Convert x and y to dictionary, whose key is the index string, and value
    # # is the value of x and y.
    # x_dictionary = dict([(str(index), value) for index, value in x.items()])
    # y_dictionary = dict([(str(index), value) for index, value in y.items()])
    #
    # # List of index to be shuffled.
    # z = list(x_dictionary.keys())
    # np.random.shuffle(z)
    #
    # shuffled_x_dict = {}
    # shuffled_y_dict = {}
    #
    # for i in z:
    #     key_i = str(i)
    #     shuffled_x_dict[key_i] = x_dictionary[key_i]
    #     shuffled_y_dict[key_i] = y_dictionary[key_i]
    #
    # # Convert dictionaries to pandans Series.
    # shuffled_x_dict_series = pd.Series(shuffled_x_dict)
    # shuffled_y_dict_series = pd.Series(shuffled_y_dict)
    #
    # return shuffled_x_dict_series, shuffled_y_dict_series

    print (x.shape, y.shape)
    stacked = np.hstack((np.matrix(x), y))
    np.random.shuffle(stacked)
    xi = np.array(stacked[:, 0]).flatten()
    yi = np.array(stacked[:, 1:])

    return xi, yi


def model(filter_kernels, dense_outputs, maxlen, vocab_size, nb_filter,
          cat_output):                                  # For Character Embedding use this model instead of above model
    d = 300  # Embedding Size
    Embedding_layer = Embedding(vocab_size+1, d, input_length=maxlen)
    inputs = Input(shape=(maxlen,), name='input', dtype='float32')
    embed = Embedding_layer(inputs)
    conv = Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[0],
                         border_mode='valid', activation='relu',
                         input_shape=(maxlen, d))(embed)
    conv = MaxPooling1D(pool_length=3)(conv)

    conv1 = Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[1],
                          border_mode='valid', activation='relu')(conv)
    conv1 = MaxPooling1D(pool_length=3)(conv1)

    conv2 = Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[2],
                          border_mode='valid', activation='relu')(conv1)

    # conv3 = Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[3],
    #                       border_mode='valid', activation='relu')(conv2)
    #
    # conv4 = Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[4],
    #                       border_mode='valid', activation='relu')(conv3)
    #
    # conv5 = Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[5],
    #                       border_mode='valid', activation='relu')(conv4)
    conv5 = MaxPooling1D(pool_length=3)(conv2)
    conv = Flatten()(conv)

    # Two dense layers with dropout of .5
    z = Dropout(0.5)(Dense(dense_outputs, activation='relu')(conv))
    # z = Dropout(0.5)(Dense(dense_outputs, activation='relu')(z))

    pred = Dense(cat_output, activation='softmax', name='output')(z)

    model = Model(input=inputs, output=pred)

    sgd = SGD(lr=0.001, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=sgd,
                  metrics=['accuracy'])

    return model


def preprocess(df):
    result_df = df.replace(
        to_replace='\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', value='@url', regex=True)
    result_df = result_df.replace(to_replace=' ', value='_', regex=True)

    X = result_df['Text']

    le = LabelEncoder()
    enc = LabelBinarizer()

    result_df['ID'] = le.fit_transform(result_df['ID'])
    enc.fit(list(result_df['ID']))

    Y = enc.transform(list(result_df['ID']))

    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=1)

    print(y_train)

    return X_train, X_test, y_train, y_test


print("reading file")
raw_train_df = pd.read_csv(
    SML_TRAIN_FILE, delimiter='\t', header=None, names=['ID', 'Text'])

print("preprocessing")
X_train, X_test, y_train, y_test = preprocess(raw_train_df)

vocab, reverse_vocab, vocab_size, check = create_vocab_set()

print("building model...")
model = model(filter_kernels, dense_outputs, maxlen, vocab_size,
              nb_filter, cat_output)

print("fitting model...")
initial = datetime.datetime.now()

for e in range(nb_epoch):
    # xi, yi = shuffle_matrix(X_train, y_train)
    # xi_test, yi_test = shuffle_matrix(X_test, y_test)

    xi, yi = X_train, y_train
    xi_test, yi_test = X_test, y_test

    batches = mini_batch_generator(xi, yi, vocab, vocab_size,
                                   check, maxlen,
                                   batch_size=batch_size)

    test_batches = mini_batch_generator(xi_test, yi_test, vocab,
                                        vocab_size, check, maxlen,
                                        batch_size=batch_size)

    accuracy = 0.0
    loss = 0.0
    step = 1
    start = datetime.datetime.now()
    print('Epoch: {}'.format(e))
    for x_train, y_train_ in batches:
        f = model.train_on_batch(x_train, y_train_)
        loss += f[0]
        loss_avg = loss / step
        accuracy += f[1]
        accuracy_avg = accuracy / step
        if step % 100 == 0:
            print('  Step: {}'.format(step))
            print('\tLoss: {}. Accuracy: {}'.format(loss_avg, accuracy_avg))
        step += 1

    test_accuracy = 0.0
    test_loss = 0.0
    test_step = 1

    for x_test_batch, y_test_batch in test_batches:
        f_ev = model.test_on_batch(x_test_batch, y_test_batch)
        test_loss += f_ev[0]
        test_loss_avg = test_loss / test_step
        test_accuracy += f_ev[1]
        test_accuracy_avg = test_accuracy / test_step
        test_step += 1
    stop = datetime.datetime.now()
    e_elap = stop - start
    t_elap = stop - initial
    print(
        'Epoch {}. Loss: {}. Accuracy: {}\nEpoch time: {}. Total time: {}\n'.format(e, test_loss_avg, test_accuracy_avg,
                                                                                    e_elap, t_elap))
