[1]: http://cs229.stanford.edu/proj2017/final-reports/5241953.pdf
[2]: https://www.aclweb.org/anthology/E17-2106
[Adam]: https://arxiv.org/pdf/1412.6980.pdf
[VADER]: http://datameetsmedia.com/vader-sentiment-analysis-explained/
[Penn Treebank]: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.9.8216&rep=rep1&type=pdf
[Glove]: https://nlp.stanford.edu/projects/glove/
[NLTK_DATA]: http://www.nltk.org/nltk_data/



## Literature Review of Project 1 COMP90051

NB: Naive Bayes; SVM: Support Vector Machine; NN: Neural Network; CNN: Convolutional NN

| Literature                    | Resources                                               | Preprocessing                                                | Feature Engineering                                          | Models                                                       |
| ----------------------------- | ------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [1], 2017<br />(Student work) | Politicians tweets<br />1,243,370 tweets<br />545 users | 1. filter out retweet data<br />2. remove links, numbers, special chars except @ and #.<br />3. tokenize<br />4. remove stop words<br />5. Lemmatize | 1. Bag of words: unigram<br />2. part of speech: [Penn Treebank] P.O.S. tagset [NLTK_DATA] to assign **non-lemmatize** word<br />3. Sentiment: apply to **nontokenize** tweet using [VADER]<br />4. Word2vec: pretrained model from [Glove] | 1. Multinomial NB with laplace<br />2. SVM (linear or Gaussian)<br />3. NN(fully-connected feedforward): [Adam] optimization |
| [2], 2017                     | 9000 users<br />9,000,000 tweets                        | Normalized numbers, names, urls                              | 1. Character n-gram (1, **2**, 3)                            | CNN:<br />1. Character embedding module:<br />2. Convolution module:<br />3. Softmax module: |
|                               |                                                         |                                                              |                                                              |                                                              |
|                               |                                                         |                                                              |                                                              |                                                              |
|                               |                                                         |                                                              |                                                              |                                                              |
|                               |                                                         |                                                              |                                                              |                                                              |
|                               |                                                         |                                                              |                                                              |                                                              |
|                               |                                                         |                                                              |                                                              |                                                              |
|                               |                                                         |                                                              |                                                              |                                                              |

Common benchmark: Naive Bayes with n-gram features [1]