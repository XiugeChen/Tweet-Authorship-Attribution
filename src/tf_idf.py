import nltk
import pandas as pd
import numpy as np

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

#nltk.download('averaged_perceptron_tagger')
#nltk.download('wordnet')
#nltk.download('stopwords')

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
    result_df = df.replace(to_replace ='@handle', value = '@handle', regex=True)
    # remove # but save tags
    result_df = result_df.replace(to_replace ='#', value = '', regex=True)
    # remove links and urls
    result_df = result_df.replace(to_replace ='\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', value = '@url', regex=True)
    
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
def preprocess(df, rmv_rt=True, rmv_all_spec=False, rmv_stop=False, lemmatize=False, word_ngram=1, add_pos=False, pos_ngram=1):
    logging.info('Preprocess starting')
    
    if rmv_rt:
        df = filter_RT(df)
    else:
        df = df.replace(to_replace ='RT @handle', value = '@RT @handle', regex=True)
    
    result_df = rmv_special_term(df, rmv_all_spec)
    
    result_df['Text'] = result_df['Text'].apply(lambda x: x.lower().rstrip().lstrip())
    
    # tokenize sentence
    tknzr = TweetTokenizer()
    result_df['Text'] = result_df['Text'].apply(lambda x: tknzr.tokenize(x))
    
    # add pos tags
    if add_pos:
        result_df['Text'] = result_df['Text'].apply(lambda x: pos_analysis(x, ngram=pos_ngram))
        
    if word_ngram > 1:
        result_df['Text'] = result_df['Text'].apply(lambda x: get_word_ngram(x, ngram=word_ngram))
         
    # remove stop words
    if rmv_stop:
        result_df['Text'] = result_df['Text'].apply(lambda x: [i for i in x if i not in cached_stop_words])
    
    # stem words
    if lemmatize:
        result_df['Text'] = result_df['Text'].apply(lambda x: [lemmatizer.lemmatize(i, get_wordnet_pos(i)) for i in x])
    
    logging.info('Preprocess ending')  
    return result_df
    
# analysis POS (part of speech) and ngram
def pos_analysis(word_list, ngram=1):
    postag_word = np.asarray(nltk.pos_tag(word_list))
    
    if ngram == 1:
        return postag_word.ravel();
    
    # get only pos tags
    pos_tags = []
    for chunk in postag_word:
        pos_tags.append(chunk[1])
        
    pos_ngrams = []
    if ngram >= len(pos_tags):
        pos_ngrams.append(''.join(tag for tag in pos_tags))
    else:
        for i in range(0, len(pos_tags)-ngram+1):
            new_tag = ''
            for j in range(i, i+ngram):
                new_tag += str(pos_tags[j])
                
            pos_ngrams.append(new_tag)
        
    result = np.append(postag_word, [i for i in pos_ngrams])
    
    return result
    
# get word ngram
def get_word_ngram(raw_word_list, ngram=1):
    if ngram == 1:
        return raw_word_list
    
    result = raw_word_list
    # get pure word list, eliminate pos tags
    pure_word_list = []
    for word in raw_word_list:
        if not word[0].isupper():
            pure_word_list.append(word)
            
    if ngram >= len(pure_word_list):
        result = np.append(result, ''.join(word for word in pure_word_list))
    else:
        for i in range(0, len(pure_word_list)-ngram+1):
            new_tag = ''
            for j in range(i, i+ngram):
                new_tag += str(pure_word_list[j])
                
            result = np.append(result, new_tag)
    
    return result
    
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize

from scipy import sparse

from gensim.models import Word2Vec
from gensim.models import KeyedVectors

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# merge all tweets from same user
def merge_tweets(df):
    df['Text'] = df['Text'].apply(lambda x: ''.join(i + ' ' for i in x))
    
    aggregation_functions = {'Text': ''.join}
    result_df = df.groupby(df['ID']).aggregate(aggregation_functions).reset_index()
    
    df['Text'] = df['Text'].apply(lambda x: x.rstrip())
    print("finish merge tweets")
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
    
    print("Finish tf-idf feature extraction")

    return tfidf_train_df, tfidf_test_df

# add lexicon features
def add_lexicon_features(df, feature_vec=None):
    features = []
    for index, row in df.iterrows():
        text, feature = row['Text'], []
        feature.extend(avg_var_word_len(text))
        feature.append(len(text))
        
        features.append(feature)
    
    A = np.array(features)
    A = normalize(A, axis=0, norm='max')
    
    if feature_vec == None:
        print("finish add lexicon features")
        return sparse.csr_matrix(A)
    else:
        for column in A.T: 
            feature_vec = sparse.hstack((feature_vec, column[:,None]))
        
        print("finish add lexicon features")
        return feature_vec

# caculate averge word length for given sentence (word should starts with alphabet letters)
def avg_var_word_len(text):
    words, length = text.split(' '), []
    
    for word in words:
        if word[0].isalpha() and word[0].islower():
            length.append(len(word))
            
    length = np.array(length)
    
    if length.size == 0:
        return [0, 0, 0]
    
    return [np.mean(length), np.std(length), np.median(length)]
    
# generate substring for df
def generate_substring(df, length=6):
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
    
    print("Finish substring extraction")
    return df
    
analyser = SentimentIntensityAnalyzer()

# sentiment analysis
def sentiment_analysis(df, feature_vec=None):
    sentiment_scores = []
    for index, row in df.iterrows():
        text, ori_words = row['Text'], []
        words = text.split(' ')
        
        for word in words:
            if '@rt' in word:
                break
                
            if len(word) < 1:
                continue
                
            if not word[0].isupper():
                ori_words.append(word)
                
        sentence = ''.join(i + ' ' for i in ori_words).rstrip()
        score = analyser.polarity_scores(sentence)
        sentiment_scores.append([score.get('neg'), score.get('neu'), score.get('pos'), score.get('compound')])
        
    A = np.array(sentiment_scores)
    if feature_vec == None:
        print("finish add sentiment features")
        return sparse.csr_matrix(A)
    else:
        for column in A.T: 
            feature_vec = sparse.hstack((feature_vec, column[:,None]))
        
        print("finish add sentiment features")
        return feature_vec
    
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

import csv

# models
models = [svm.LinearSVC(C=0.68, max_iter=1000)]
          #LogisticRegression(solver="lbfgs", penalty="l2", C=0.65, multi_class='auto', max_iter=1000)]
         # MultinomialNB()]

titles = ['LinearSVM, 0.68']
         #'LogisticRegression']
          #'MNB']
          
def cross_validate_tf_idf(df, merge=False, add_lexicon=False, substring=False, substring_len=3, add_sentiment=False):
    cv = KFold(n_splits=10, random_state=90051, shuffle=True)
    
    scores = {}
    for train_index, test_index in cv.split(df):
        train_df, test_df = df.iloc[train_index].reset_index(drop=True), df.iloc[test_index].reset_index(drop=True)
        
        # merge all tweets from same user to one document string
        if merge:
            train_df = merge_tweets(train_df)
        else:
            train_df['Text'] = train_df['Text'].apply(lambda x: ''.join(i + ' ' for i in x).rstrip())
        
        test_df['Text'] = test_df['Text'].apply(lambda x: ''.join(i + ' ' for i in x).rstrip())
        
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
        
        for title, model in zip(titles, models):
            model.fit(X_train, y_train)
            predicted_labels = model.predict(X_test)
            acc = accuracy_score(predicted_labels, y_test)
            
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
        train_df['Text'] = train_df['Text'].apply(lambda x: ''.join(i + ' ' for i in x).rstrip())
        
    test_df['Text'] = test_df['Text'].apply(lambda x: ''.join(i + ' ' for i in x).rstrip())
    
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
    
    print("start train model")
    for title, model in zip(titles, models):
        model.fit(X_train, y_train)
        print("model fit finished")
        label = model.predict(X_test)
        print("model predict finished")
    
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

print("Finish reading")
preprocess_train_df = preprocess(raw_train_df, rmv_rt=False, rmv_all_spec=False, rmv_stop=False, lemmatize=False, word_ngram=1, add_pos=True, pos_ngram=1)
preprocess_test_df = preprocess(raw_test_df, rmv_rt=False, rmv_all_spec=False, rmv_stop=False, lemmatize=False, word_ngram=1, add_pos=True, pos_ngram=1)
#print(preprocess_train_df.shape, preprocess_test_df.shape)

print("Finsih preprocess")

#cross_validate_tf_idf(preprocess_train_df, merge=False, add_lexicon=False, substring=False, substring_len=3, add_sentiment=True)

predict_tf_idf(preprocess_train_df, preprocess_test_df, merge=False, add_lexicon=False, substring=False, substring_len=3, add_sentiment=True)
print("finished")
