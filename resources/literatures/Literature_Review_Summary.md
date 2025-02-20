[1]: http://cs229.stanford.edu/proj2017/final-reports/5241953.pdf
[2]: https://www.aclweb.org/anthology/E17-2106
[3]: https://arxiv.org/pdf/1902.09723.pdf
[Adam]: https://arxiv.org/pdf/1412.6980.pdf
[VADER]: http://datameetsmedia.com/vader-sentiment-analysis-explained/
[Penn Treebank]: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.9.8216&rep=rep1&type=pdf
[Glove]: https://nlp.stanford.edu/projects/glove/
[NLTK_DATA]: http://www.nltk.org/nltk_data/



## Literature Review of Project 1 COMP90051

### Shorthands used
- NB: Naive Bayes
- SVM: Support Vector Machine
- NN: Neural Network
    - CNN: Convolutional NN
    - RNN: Recurrent NN

| Literature                    | Resources                                               | Preprocessing                                                | Feature Engineering                                          | Models                                                       |
| ----------------------------- | ------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [1], 2017<br />(Student work) | Politicians tweets<br />1,243,370 tweets<br />545 users | 1. filter out retweet data<br />2. remove links, numbers, special chars except @ and #.<br />3. tokenize<br />4. remove stop words<br />5. Lemmatize | 1. Bag of words: unigram<br />2. part of speech: [Penn Treebank] P.O.S. tagset [NLTK_DATA] to assign **non-lemmatize** word<br />3. Sentiment: apply to **nontokenize** tweet using [VADER]<br />4. Word2vec: pretrained model from [Glove] | 1. Multinomial NB with laplace<br />2. SVM (linear or Gaussian)<br />3. NN(fully-connected feedforward): [Adam] optimization |
| [2], 2017                     | 9000 users<br />9,000,000 tweets                        | Normalized numbers, names, urls                              | 1. Character n-gram (1, **2**, 3)                            | CNN:<br />1. Character embedding module:<br />2. Convolution module:<br />3. Softmax module: |
| [3], 2019                     | PAN 2012 Task 1<br />14 authors                         |                                                              | POS                                                          | RNN:<br />1. POS encoder (CNN/LSTM): Syntactic representation<br />2. Sentence encoder(LSTM)<br />3. Attention layer: Document representation, reward sentence<br />4. Softmax<br /> |
| [Machine Learning approach to authorship attribution of literary texts](https://pdfs.semanticscholar.org/d2a5/a2326ed00ef2ed4ba9a4a2dff3ce9f765a28.pdf) | 14,000 users |1. At preprocessor: Replace all instances of @username with @ <br /> 2. Hash preprocessor: Replace #tag with # <br /> 3. Both preprocessor: Both At and Hash applied | Source Code Attribute Profile (SCAP) |                                                              |
|                               |                                                         |                                                              |                                                              |                                                              |
|                               |                                                         |                                                              |                                                              |                                                              |
|                               |                                                         |                                                              |                                                              |                                                              |
|                               |                                                         |                                                              |                                                              |                                                              |
|                               |                                                         |                                                              |                                                              |                                                              |

Common benchmark: Naive Bayes with n-gram features [1]