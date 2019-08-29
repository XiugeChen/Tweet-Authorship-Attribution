import nltk
import pandas as pd
import logging
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet

TRAIN_FILE = "../resources/data/train_tweets.txt"
SML_TRAIN_FILE = "../resources/data/test.txt"
TEST_FILE = "../resources/data/test_tweets_unlabeled.txt"

GLOVE_25D = "../resources/glove/glove.twitter.27B.25d.txt"
GLOVE_200D = "../resources/glove/glove.twitter.27B.200d.txt"

# remove all RTs (reweets)
def filter_RT(df):
    rt = df['Text'].str.startswith('RT @handle')
    not_rt = [not i for i in rt]
    result_df = df[not_rt]
    result_df = result_df.reset_index(drop=True)
    return result_df
    
# remove special terms like "@handle", links
def rmv_special_term(df, rmv_all_spec=False):
    # remove @s
    result_df = df.replace(to_replace ='@handle', value = '', regex=True)
    # remove # but save tags
    result_df = result_df.replace(to_replace ='#', value = '', regex=True)
    # remove links and urls
    result_df = result_df.replace(to_replace ='\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', value = '', regex=True)
    
    # filter out all chars except 1-9/a-z/A-Z, such as :-( ' , . / \ 
    if rmv_all_spec:
        result_df = result_df.replace(to_replace ='([^0-9A-Za-z \t])|(\w+:\/\/\S+)', value = '', regex=True)
        
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
def preprocess(df, rmv_rt=True, rmv_all_spec=False, rmv_stop=False, lemmatize=False):
    logging.info('Preprocess starting')
    
    if rmv_rt:
        df = filter_RT(df)
    
    result_df = rmv_special_term(df, rmv_all_spec)
    
    #result_df['Text'] = result_df['Text'].str.lower()
    result_df['Text'] = result_df['Text'].apply(lambda x: x.lower().rstrip().lstrip())
    
    # tokenize sentence
    tknzr = TweetTokenizer()
    result_df['Text'] = result_df['Text'].apply(lambda x: tknzr.tokenize(x))
        
    # remove stop words
    if rmv_stop:
        result_df['Text'] = result_df['Text'].apply(lambda x: [i for i in x if i not in cached_stop_words])
    
    # stem words
    if lemmatize:
        result_df['Text'] = result_df['Text'].apply(lambda x: [lemmatizer.lemmatize(i, get_wordnet_pos(i)) for i in x])
    
    logging.info('Preprocess ending')  
    return result_df
    
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from gensim.models import Word2Vec
from gensim.models import KeyedVectors

# merge all tweets from same user
def merge_tweets(df):
    aggregation_functions = {'Text': 'sum'}
    result_df = df.groupby(df['ID']).aggregate(aggregation_functions).reset_index()
    
    return result_df
    
# extract tf-idf features
def tf_idf(train_df, test_df):
    #get the text column 
    train_docs = train_df['Text'].tolist()
    test_docs = test_df['Text'].tolist()
    
    #create a vocabulary of words, 
    #ignore words that appear in max_df of documents
    stop_words = stopwords.words('english')
    cv = CountVectorizer(max_df=0.85, stop_words=stop_words, decode_error='ignore')
    trian_wc_vec = cv.fit_transform(train_docs)
    test_wc_vec = cv.transform(test_docs)
    
    # get tfidf
    transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_train_df = transformer.fit_transform(trian_wc_vec)
    tfidf_test_df = transformer.transform(test_wc_vec)

    return tfidf_train_df, tfidf_test_df
    
# extract word vector features
def wordvec(train_df, test_df, sizeD=25):
    return
    
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

import csv

# models
models = [svm.LinearSVC(C=1, max_iter=1000),
          MultinomialNB(),
          ]

titles = ['LinearSVM',
          'MNB']
          
def cross_validate_tf_idf(df):
    cv = KFold(n_splits=10, random_state=90051, shuffle=True)
    
    scores = {}
    for train_index, test_index in cv.split(df):
        train_df, test_df = df.iloc[train_index].reset_index(drop=True), df.iloc[test_index].reset_index(drop=True)
        
        # merge all tweets from same user to one document string
        train_df = merge_tweets(train_df)
        
        train_df['Text'] = train_df['Text'].apply(lambda x: ''.join(i + ' ' for i in x).rstrip())
        test_df['Text'] = test_df['Text'].apply(lambda x: ''.join(i + ' ' for i in x).rstrip())
        
        X_train, X_test = tf_idf(train_df, test_df)
        y_train, y_test = train_df['ID'], test_df['ID']
        
        for title, model in zip(titles, models):
            model.fit(X_train, y_train)
            acc = accuracy_score(model.predict(X_test), y_test)
            
            if title in scores.keys():
                scores[title] += acc
            else:
                scores[title] = acc
        
    for title in titles:
        acc = scores[title] / 10
        print("####INFO: trainning", title, acc)
        
def predict_tf_idf(train_df, test_df):
    # merge all tweets from same user to one document string
    train_df = merge_tweets(train_df)
        
    train_df['Text'] = train_df['Text'].apply(lambda x: ''.join(i + ' ' for i in x).rstrip())
    test_df['Text'] = test_df['Text'].apply(lambda x: ''.join(i + ' ' for i in x).rstrip())
        
    X_train, X_test = tf_idf(train_df, test_df)
    y_train = train_df['ID']
    
    for title, model in zip(titles, models):
        model.fit(X_train, y_train)
        label = model.predict(X_test)
    
        wtr = csv.writer(open('prediction_' + title + '.csv', 'w'), delimiter=',', lineterminator='\n')
        wtr.writerow(['Id','Predicted'])
        for i in range(0, label.size):
            wtr.writerow([i+1, label[i]])
    
    return
    
# test
pd.options.mode.chained_assignment = None

raw_train_df = pd.read_csv(TRAIN_FILE, delimiter='\t', header=None, names=['ID','Text'])
raw_test_df = pd.read_csv(TEST_FILE, delimiter='\t', header=None, names=['Text'])
# print(train_df.shape)
# print(test_df.shape)

preprocess_train_df = preprocess(raw_train_df, rmv_rt=True, rmv_all_spec=False, rmv_stop=True, lemmatize=True)
preprocess_test_df = preprocess(raw_test_df, rmv_rt=False, rmv_all_spec=False, rmv_stop=True, lemmatize=True)
#print(preprocess_train_df.shape, preprocess_test_df.shape)

'''
merged_train_df = merge_tweets(preprocess_train_df)
merged_test_df = preprocess_test_df
merged_train_df['Text'] = merged_train_df['Text'].apply(lambda x: ''.join(i + ' ' for i in x).rstrip())
merged_test_df['Text'] = merged_test_df['Text'].apply(lambda x: ''.join(i + ' ' for i in x).rstrip())

tf_idf_train, tf_idf_test = tf_idf(merged_train_df, merged_test_df)
print(tf_idf_train.shape, tf_idf_test.shape)
print(tf_idf_train)
'''

#cross_validate_tf_idf(preprocess_train_df)

predict_tf_idf(preprocess_train_df, preprocess_test_df)
print("finished")  
