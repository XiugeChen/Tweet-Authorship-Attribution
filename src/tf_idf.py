import nltk
import pandas as pd
import numpy as np

import logging
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

from scipy import sparse

from gensim.models import Word2Vec
from gensim.models import KeyedVectors

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import csv

from extract_users_from_source import extract_users_from_source

# Extract certain number of users from the source file for experiments.
extract_users_from_source(50)
extract_users_from_source(100)
extract_users_from_source(200)
extract_users_from_source(500)
extract_users_from_source(1000)

DATA_FOLDER = "../resources/data/"
TRAIN_FILE_NAME = "train_tweets.txt"

TRAIN_FILE = DATA_FOLDER + TRAIN_FILE_NAME

SML_TRAIN_FILE = DATA_FOLDER + "test.txt"
TEST_FILE = DATA_FOLDER + "test_tweets_unlabeled.txt"

GLOVE_25D = "../resources/glove/glove.twitter.27B.25d.txt"
GLOVE_200D = "../resources/glove/glove.twitter.27B.200d.txt"

# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
# nltk.download('stopwords')


user200_file = DATA_FOLDER + "200_" + TRAIN_FILE_NAME

raw_train_df = pd.read_csv(
    user200_file, delimiter='\t', header=None, names=['ID', 'Text'])

complete_train_df = pd.read_csv(
    TRAIN_FILE, delimiter='\t', header=None, names=['ID', 'Text'])

print("####INFO: Finish reading")


def calculate_class_weight(raw_train_df):
    unique_users_id = raw_train_df["ID"].unique().tolist()
    # print("raw_train_df", raw_train_df)
    # print("len(unique_users_id)", len(unique_users_id))

    raw_class_weight = {}
    for unique_user_id in unique_users_id:
        raw_class_weight[unique_user_id] = len(
            raw_train_df[raw_train_df.ID == unique_users_id[0]])

    # Total number of tweets in the dataset.
    total_tweets_num = sum(raw_class_weight.values())

    class_weight = {}

    for k, v in raw_class_weight.items():
        class_weight[k] = v*1.0/total_tweets_num

    # print(class_weight)
    # print("sum(class_weight.values())", sum(class_weight.values()))

    return class_weight


# class_weight = calculate_class_weight(raw_train_df)
# print("####INFO: Finish class weight calculation")


def filter_RT(df):
    """remove all RTs (reweets)"""
    rt = df['Text'].str.startswith('RT @handle')
    not_rt = [not i for i in rt]
    result_df = df[not_rt]
    result_df = result_df.reset_index(drop=True)
    return result_df


def rmv_special_term(df, rmv_all_spec=False):
    """remove special terms like "@handle", links"""
    # remove @s
    result_df = df.replace(to_replace='@handle', value='@handle', regex=True)
    # remove # but save tags
    result_df = result_df.replace(to_replace='#', value='#', regex=True)
    # remove links and urls
    result_df = result_df.replace(
        to_replace='\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', value='@url', regex=True)

    # filter out all chars except 1-9/a-z/A-Z, such as :-( ' , . / \
    if rmv_all_spec:
        result_df = result_df.replace(
            to_replace='([^0-9A-Za-z \t])|(\w+:\/\/\S+)', value='', regex=True)

    return result_df


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


cached_stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# main call


def preprocess(df, rmv_rt=True, rmv_all_spec=False, rmv_stop=False, lemmatize=False, word_ngram=[1], add_pos=False, pos_ngram=[1]):
    logging.info('Preprocess starting')

    if rmv_rt:
        df = filter_RT(df)
    else:
        df = df.replace(to_replace='RT @handle',
                        value='@RT @handle', regex=True)

    result_df = rmv_special_term(df, rmv_all_spec)

    result_df['Text'] = result_df['Text'].apply(lambda x: x.rstrip().lstrip())

    # tokenize sentence
    tknzr = TweetTokenizer()
    result_df['Text'] = result_df['Text'].apply(lambda x: tknzr.tokenize(x))

    # add pos tags
    if add_pos:
        result_df['Text'] = result_df['Text'].apply(
            lambda x: pos_analysis(x, ngram=pos_ngram))

    if len(word_ngram) > 0:
        result_df['Text'] = result_df['Text'].apply(
            lambda x: get_word_ngram(x, ngram=word_ngram))

    # remove stop words
    if rmv_stop:
        result_df['Text'] = result_df['Text'].apply(
            lambda x: [i for i in x if i not in cached_stop_words])

    # stem words
    if lemmatize:
        result_df['Text'] = result_df['Text'].apply(
            lambda x: [lemmatizer.lemmatize(i, get_wordnet_pos(i)) for i in x])

    logging.info('Preprocess ending')
    return result_df

# analysis POS (part of speech) and ngram


def pos_analysis(word_list, ngram=[1]):
    if 0 in ngram:
        return word_list

    postag_word = np.asarray(nltk.pos_tag(word_list))

    # get only pos tags
    pos_tags = []
    for chunk in postag_word:
        pos_tags.append(chunk[1])

    result = word_list
    for n in ngram:
        pos_ngrams = []

        if n == 1:
            pos_ngrams = pos_tags
        else:
            if n >= len(pos_tags):
                pos_ngrams.append(''.join(tag for tag in pos_tags))
            else:
                for i in range(0, len(pos_tags)-n+1):
                    new_tag = ''
                    for j in range(i, i+n):
                        new_tag += str(pos_tags[j])

                    pos_ngrams.append(new_tag)

        result = np.append(result, [i for i in pos_ngrams])

    return result

# get word ngram


def get_word_ngram(raw_word_list, ngram=[1]):
    result = []
    # get pure word list, eliminate pos tags
    pure_word_list = []
    for word in raw_word_list:
        if len(word) < 1:
            continue

        if not word[0].isupper():
            pure_word_list.append(word)
        else:
            result.append(word)

    if 0 in ngram:
        return result

    for n in ngram:
        if n == 1:
            result = np.append(result, pure_word_list)
        else:
            if n >= len(pure_word_list):
                result = np.append(result, ''.join(
                    word for word in pure_word_list))
            else:
                for i in range(0, len(pure_word_list)-n+1):
                    new_tag = ''
                    for j in range(i, i+n):
                        new_tag += str(pure_word_list[j])

                    result = np.append(result, new_tag)

    return result


def merge_tweets(df):
    """merge all tweets from same user"""
    df['Text'] = df['Text'].apply(lambda x: ''.join(i + ' ' for i in x))

    aggregation_functions = {'Text': ''.join}
    result_df = df.groupby(df['ID']).aggregate(
        aggregation_functions).reset_index()

    df['Text'] = df['Text'].apply(lambda x: x.rstrip())
    #print("finish merge tweets")
    return result_df


def tf_idf(train_df, test_df, min_df=1):
    """extract tf-idf features"""
    # get the text column
    train_docs = train_df['Text'].tolist()
    test_docs = test_df['Text'].tolist()

    # create a vocabulary of words,
    # ignore words that appear in max_df of documents
    cv = CountVectorizer(max_df=0.42, min_df=min_df, decode_error='ignore')
    trian_wc_vec = cv.fit_transform(train_docs)
    test_wc_vec = cv.transform(test_docs)

    # get tfidf
    transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_train_df = transformer.fit_transform(trian_wc_vec)
    tfidf_test_df = transformer.transform(test_wc_vec)

    #print("Finish tf-idf feature extraction")

    return tfidf_train_df, tfidf_test_df


def add_lexicon_features(df, feature_vec=None):
    """add lexicon features"""
    features = []
    for index, row in df.iterrows():
        text, feature = row['Text'], []
        feature.extend(avg_var_word_len(text))
        feature.append(len(text))

        features.append(feature)

    A = np.array(features)
    A = normalize(A, axis=0, norm='max')

    if feature_vec == None:
        #print("finish add lexicon features")
        return sparse.csr_matrix(A)
    else:
        for column in A.T:
            feature_vec = sparse.hstack((feature_vec, column[:, None]))

        #print("finish add lexicon features")
        return feature_vec


def avg_var_word_len(text):
    """caculate averge word length for given sentence (word should starts 
    with alphabet letters)
    """
    words, length = text.split(' '), []

    for word in words:
        if word[0].isalpha() and word[0].islower():
            length.append(len(word))

    length = np.array(length)

    if length.size == 0:
        return [0, 0, 0]

    return [np.mean(length), np.std(length), np.median(length)]


def generate_substring(df, length=6):
    """generate substring for df"""
    iter_df = df.copy()

    for index, row in iter_df.iterrows():
        text, new_words = row['Text'], []
        words = text.split(' ')

        for word in words:
            if len(word) >= length:
                for i in range(0, len(word)-length+1):
                    sub_word = word[i:i + length]
                    new_words.append(sub_word)

            new_words.append(word)

        new_text = ''.join(i + ' ' for i in new_words).rstrip()

        df.loc[index, 'Text'] = new_text

    #print("Finish substring extraction")
    return df


analyser = SentimentIntensityAnalyzer()


def sentiment_analysis(df, feature_vec=None):
    """sentiment analysis"""
    sentiment_scores = []
    for index, row in df.iterrows():
        text, ori_words = row['Text'], []
        words = text.split(' ')

        for word in words:
            if '@rt' in word:
                break

            if len(word) < 1:
                continue

            ori_words.append(word)

        sentence = ''.join(i + ' ' for i in ori_words).rstrip()
        score = analyser.polarity_scores(sentence)
        sentiment_scores.append([score.get('neg'), score.get(
            'neu'), score.get('pos'), score.get('compound')])

    A = np.array(sentiment_scores)
    if feature_vec == None:
        #print("finish add sentiment features")
        return sparse.csr_matrix(A)
    else:
        for column in A.T:
            feature_vec = sparse.hstack((feature_vec, column[:, None]))

        #print("finish add sentiment features")
        return feature_vec


def cross_validate_tf_idf(df, merge=False, add_lexicon=False, substring=False, substring_len=3, add_sentiment=False, pca=False):
    cv = KFold(n_splits=10, random_state=90051, shuffle=True)

    scores = {}
    for train_index, test_index in cv.split(df):
        #print("Batch start")
        train_df, test_df = df.iloc[train_index].reset_index(
            drop=True), df.iloc[test_index].reset_index(drop=True)

        # merge all tweets from same user to one document string
        if merge:
            train_df = merge_tweets(train_df)
        else:
            train_df['Text'] = train_df['Text'].apply(
                lambda x: ''.join(i + ' ' for i in x).rstrip())

        test_df['Text'] = test_df['Text'].apply(
            lambda x: ''.join(i + ' ' for i in x).rstrip())

        if substring:
            train_df = generate_substring(train_df, length=substring_len)
            test_df = generate_substring(test_df, length=substring_len)

        X_train, X_test = tf_idf(train_df, test_df)
        y_train, y_test = train_df['ID'], test_df['ID']

        if add_lexicon:
            X_train = add_lexicon_features(train_df, feature_vec=X_train)
            X_test = add_lexicon_features(test_df, feature_vec=X_test)

        if add_sentiment:
            X_train = sentiment_analysis(train_df, feature_vec=X_train)
            X_test = sentiment_analysis(test_df, feature_vec=X_test)

        if pca:
            X_train, X_test = pca_reduction(X_train, X_test)

        # print("X_train.shape")
        # print(X_train.shape)

        # print("y_train.shape")
        # print(y_train.shape)

        # print("y_train.unique()")
        # print(y_train.unique())

        # print("len(y_train.unique())", len(y_train.unique()))

        for title, model in zip(titles, models):
            #print("Start train " + title)

            model.fit(X_train, y_train)
            predicted_labels = model.predict(X_test)
            acc = accuracy_score(predicted_labels, y_test)
            #print(title + " " + str(acc))

            # uncomment to print miss labeled data
            # for i in range(0, len(predicted_labels)):
            #   if predicted_labels[i] != y_test[i]:
            #        print("#" + str(i) + "; T: " + str(y_test[i]) + "; F: " + str(predicted_labels[i]) + "; Text: " + test_df.loc[i,'Text'])

            if title in scores.keys():
                scores[title] += acc
            else:
                scores[title] = acc

    for title in titles:
        acc = scores[title] / 10
        print("####INFO: trainning", title, acc)


def predict_tf_idf(train_df, test_df, merge=False, add_lexicon=False, substring=False, substring_len=3, add_sentiment=False):
    # merge all tweets from same user to one document string
    if merge:
        train_df = merge_tweets(train_df)
    else:
        train_df['Text'] = train_df['Text'].apply(
            lambda x: ''.join(i + ' ' for i in x).rstrip())

    test_df['Text'] = test_df['Text'].apply(
        lambda x: ''.join(i + ' ' for i in x).rstrip())

    if substring:
        train_df = generate_substring(train_df, length=substring_len)
        test_df = generate_substring(test_df, length=substring_len)

    X_train, X_test = tf_idf(train_df, test_df)
    y_train = train_df['ID']

    if add_lexicon:
        X_train = add_lexicon_features(train_df, feature_vec=X_train)
        X_test = add_lexicon_features(test_df, feature_vec=X_test)

    if add_sentiment:
        X_train = sentiment_analysis(train_df, feature_vec=X_train)
        X_test = sentiment_analysis(test_df, feature_vec=X_test)

    for title, model in zip(titles, models):
        #print("start training")
        model.fit(X_train, y_train)
        #print("model " + title + " fit finished")
        label = model.predict(X_test)
        #print("model predict finished")

        wtr = csv.writer(open('prediction_' + title + '.csv',
                              'w'), delimiter=',', lineterminator='\n')
        wtr.writerow(['Id', 'Predicted'])
        for i in range(0, label.size):
            wtr.writerow([i+1, label[i]])

    return

# Helper functions for stacking


def predict(model, train_df, test_df, wordngram=[1], pos=False, posngram=[1], addsentiment=True, min_tf_idf=1):
    train_df = preprocess(train_df, rmv_rt=False, rmv_all_spec=False, rmv_stop=False,
                          lemmatize=False, word_ngram=wordngram, add_pos=pos, pos_ngram=posngram)
    test_df = preprocess(test_df, rmv_rt=False, rmv_all_spec=False, rmv_stop=False,
                         lemmatize=False, word_ngram=wordngram, add_pos=pos, pos_ngram=posngram)

    train_df['Text'] = train_df['Text'].apply(
        lambda x: ''.join(i + ' ' for i in x).rstrip())
    test_df['Text'] = test_df['Text'].apply(
        lambda x: ''.join(i + ' ' for i in x).rstrip())

    X_train, X_test = tf_idf(train_df, test_df, min_df=min_tf_idf)
    y_train, y_test = train_df['ID'], test_df['ID']

    if addsentiment:
        X_train = sentiment_analysis(train_df, feature_vec=X_train)
        X_test = sentiment_analysis(test_df, feature_vec=X_test)

    model.fit(X_train, y_train)
    train_labels = model.predict(X_train)
    predicted_labels = model.predict(X_test)

    return np.array(train_labels).reshape(-1, 1), np.array(predicted_labels).reshape(-1, 1)


def stacking_cross_validate(raw_df, add_sentiment=True):
    cv = KFold(n_splits=10, random_state=90051, shuffle=True)

    score = 0
    for train_index, test_index in cv.split(raw_df):
        train_df, test_df = raw_df.iloc[train_index].reset_index(
            drop=True), raw_df.iloc[test_index].reset_index(drop=True)
        y_train, y_test = train_df['ID'], test_df['ID']

        svm_model = svm.LinearSVC(C=0.68, max_iter=1000)

        train_1, test_1 = predict(svm_model, train_df, test_df, wordngram=[
                                  1], pos=True, posngram=[1], addsentiment=True, min_tf_idf=1)
        train_2, test_2 = predict(svm_model, train_df, test_df, wordngram=[
                                  1, 2], pos=False, posngram=[1], addsentiment=True, min_tf_idf=1)
        train_3, test_3 = predict(svm_model, train_df, test_df, wordngram=[
                                  1], pos=True, posngram=[1000], addsentiment=True, min_tf_idf=1)

        h_model = svm.LinearSVC(C=0.68, max_iter=1000)

        X_train, X_test = [], []
        for i in range(0, len(train_1)):
            X_train.append(str(train_1[i]) + ' ' +
                           str(train_2[i]) + ' ' + str(train_3[i]))
        for i in range(0, len(test_1)):
            X_test.append(str(test_1[i]) + ' ' +
                          str(test_2[i]) + ' ' + str(test_3[i]))

        X_train = np.array(X_train)
        X_test = np.array(X_test)

        stop_words = stopwords.words('english')
        cv = CountVectorizer(
            max_df=0.85, stop_words=stop_words, decode_error='ignore')
        trian_wc_vec = cv.fit_transform(X_train)
        test_wc_vec = cv.transform(X_test)

        # get tfidf
        transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
        X_train = transformer.fit_transform(trian_wc_vec)
        X_test = transformer.transform(test_wc_vec)

        h_model.fit(X_train, y_train)
        train_acc = accuracy_score(h_model.predict(X_train), y_train)

        predicted_labels = h_model.predict(X_test)
        acc = accuracy_score(predicted_labels, y_test)

        sub_acc_1, sub_acc_2, sub_acc_3 = accuracy_score(train_1, y_train), accuracy_score(
            train_2, y_train), accuracy_score(train_3, y_train)
        # print("####INFO train error: ", train_acc, sub_acc_1, sub_acc_2, sub_acc_3)
        sub_acc_1, sub_acc_2, sub_acc_3 = accuracy_score(
            test_1, y_test), accuracy_score(test_2, y_test), accuracy_score(test_3, y_test)
        # print("####INFO test error: ", acc, sub_acc_1, sub_acc_2, sub_acc_3)

        # uncomment to print miss labeled data
        # for i in range(0, len(predicted_labels)):
        #   if predicted_labels[i] != y_test[i]:
        #        print("#" + str(i) + "; T: " + str(y_test[i]) + "; F: " + str(predicted_labels[i]) + "; Text: " + test_df.loc[i,'Text'])

        score += acc

    avg_acc = score / 10
    print("####INFO: trainning", 'Stacking', avg_acc)


# test
pd.options.mode.chained_assignment = None

# raw_train_df = pd.read_csv(SML_TRAIN_FILE, delimiter='\t', header=None, names=['ID','Text'])
# raw_test_df = pd.read_csv(TEST_FILE, delimiter='\t', header=None, names=['Text'])
# print(train_df.shape)
# print(test_df.shape)

# print("Finish reading")
# preprocess_train_df = preprocess(raw_train_df, rmv_rt=False, rmv_all_spec=False, rmv_stop=False, lemmatize=False, word_ngram=[1], add_pos=True, pos_ngram=[1])
# preprocess_test_df = preprocess(raw_test_df, rmv_rt=False, rmv_all_spec=False, rmv_stop=False, lemmatize=False, word_ngram=[1], add_pos=True, pos_ngram=[1])
#print(preprocess_train_df.shape, preprocess_test_df.shape)

# print("Finish preprocess")
# cross_validate_tf_idf(preprocess_train_df, merge=False, add_lexicon=False, substring=False, substring_len=3, add_sentiment=True, pca=False)

#stacking_cross_validate(raw_train_df, add_sentiment=True)

# predict_tf_idf(preprocess_train_df, preprocess_test_df, merge=False, add_lexicon=False, substring=False, substring_len=3, add_sentiment=True)
# print("finished")

"""
print("\n####INFO: Running experiment one")
# models
# models = [svm.LinearSVC(C=0.68, max_iter=1000, class_weight=class_weight),
models = [svm.LinearSVC(C=0.68, max_iter=1000),
            SGDClassifier(loss="hinge", penalty="l2",
                        max_iter=1000, n_jobs=-1, tol=1e-4),
            LogisticRegression(solver="lbfgs", penalty="l2", C=0.68,
                                multi_class='auto', max_iter=1000, fit_intercept=False),
            MultinomialNB(),
            KNeighborsClassifier(n_neighbors=5)]

titles = ['LinearSVM',
            'SGDClassifier',
            'LogisticRegression',
            'MNB',
            'KNN']

print("\n####INFO: feature combination a")
preprocess_train_df = preprocess(raw_train_df, rmv_rt=False, rmv_all_spec=False,
                                    rmv_stop=False, lemmatize=False, word_ngram=[1], add_pos=False, pos_ngram=[1])
print("####INFO: Finish preprocess")
cross_validate_tf_idf(preprocess_train_df, merge=False, add_lexicon=False,
                        substring=False, substring_len=3, add_sentiment=False, pca=False)

# create models again, as naive bayesian does not support negative number.
# models = [svm.LinearSVC(C=0.68, max_iter=1000, class_weight=class_weight),
models = [svm.LinearSVC(C=0.68, max_iter=1000),
            SGDClassifier(loss="hinge", penalty="l2",
                        max_iter=1000, n_jobs=-1, tol=1e-4),
            LogisticRegression(solver="lbfgs", penalty="l2", C=0.68,
                                multi_class='auto', max_iter=1000, fit_intercept=False),
            KNeighborsClassifier(n_neighbors=5)]

titles = ['LinearSVM',
            'SGDClassifier',
            'LogisticRegression',
            'KNN']

print("\n####INFO: feature combination b")
preprocess_train_df = preprocess(raw_train_df, rmv_rt=False, rmv_all_spec=False,
                                    rmv_stop=False, lemmatize=False, word_ngram=[1], add_pos=True, pos_ngram=[1])
print("####INFO: Finish preprocess")
cross_validate_tf_idf(preprocess_train_df, merge=False, add_lexicon=False,
                        substring=False, substring_len=3, add_sentiment=True, pca=False)

print("\n####INFO: feature combination c")
preprocess_train_df = preprocess(raw_train_df, rmv_rt=False, rmv_all_spec=False,
                                    rmv_stop=False, lemmatize=False, word_ngram=[1, 2], add_pos=True, pos_ngram=[1, 1000])
print("####INFO: Finish preprocess")
cross_validate_tf_idf(preprocess_train_df, merge=False, add_lexicon=False,
                        substring=False, substring_len=3, add_sentiment=True, pca=False)

print("\n####INFO: feature combination d")
preprocess_train_df = preprocess(raw_train_df, rmv_rt=False, rmv_all_spec=False,
                                    rmv_stop=False, lemmatize=False, word_ngram=[1, 2], add_pos=True, pos_ngram=[1, 1000])
print("####INFO: Finish preprocess")
cross_validate_tf_idf(preprocess_train_df, merge=False, add_lexicon=False,
                        substring=True, substring_len=3, add_sentiment=True, pca=False)

print("\n####INFO: feature combination e")
preprocess_train_df = preprocess(raw_train_df, rmv_rt=False, rmv_all_spec=False,
                                    rmv_stop=False, lemmatize=False, word_ngram=[1, 2], add_pos=True, pos_ngram=[1, 1000])
print("####INFO: Finish preprocess")
cross_validate_tf_idf(preprocess_train_df, merge=False, add_lexicon=True,
                        substring=True, substring_len=3, add_sentiment=True, pca=False)

stacking_cross_validate(raw_train_df, add_sentiment=True)
print("\n####INFO: Experiment one finished")
"""


print("\n####INFO: Running experiment two")

target_data_size_list = [50, 100, 200, 500, 1000]
source_file_list = []

for target_data_size in target_data_size_list:
    source_file_list.append("{0}{1}_{2}".format(
        DATA_FOLDER, target_data_size, TRAIN_FILE_NAME))

print("source_file_list")
print(source_file_list)

models = [svm.LinearSVC(C=0.68, max_iter=1000),
          SGDClassifier(loss="hinge", penalty="l2",
                        max_iter=1000, n_jobs=-1, tol=1e-4),
          LogisticRegression(solver="lbfgs", penalty="l2", C=0.68,
                             multi_class='auto', max_iter=1000, fit_intercept=False),
          KNeighborsClassifier(n_neighbors=5)]

titles = ['LinearSVM', 'SGDClassifier', 'LogisticRegression', 'KNN']

for file_name in source_file_list:
    raw_train_df = pd.read_csv(
        file_name, delimiter='\t', header=None, names=['ID', 'Text'])
    print(
        "\n####INFO: feature combination b, file name: {0}".format(file_name))
    preprocess_train_df = preprocess(raw_train_df, rmv_rt=False, rmv_all_spec=False,
                                     rmv_stop=False, lemmatize=False, word_ngram=[1], add_pos=True, pos_ngram=[1])
    print("####INFO: Finish preprocess")
    cross_validate_tf_idf(preprocess_train_df, merge=False, add_lexicon=False,
                          substring=False, substring_len=3, add_sentiment=True, pca=False)
