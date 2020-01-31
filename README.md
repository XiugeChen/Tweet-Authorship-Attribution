# Tweet-Authorship-Attribution

## Introduction

* This is the assignment 1 for [COMP90051](https://handbook.unimelb.edu.au/2018/subjects/comp90051) (Statistical Machine Learning), of the University of Melbourne, 2019 Semester 2.

* A simplification of the more general problem of authorship attribution, which automatically identifies the authorship of a document. In this project, we are given approximate 328k tweets labelled with over 10k authors.

* Using mainly n-gram, Part of Speech (POS), sentiment intensity and tf-idf features, classified using linear SVM and stacking methods.

* [Report](COMP90051_Assignment1_Report_Group199.pdf)

## Group Information
### Group Number
Group 199

### Kaggle Team Name
Jack Dorsey

### Group Memeber
- Xiuge Chen
- Shizhe Cai
- An Luo

## How to run the code
1. Make sure all dependencies (specified below) are installed
2. Move to the src folder by using `cd src`
3. Run the program using command `python tf_idf.py`, or `python3 tf_idf.py` if your python 3 is invoked using `python3`.

## Structure of the repository
### minutes
This is the meeting minutes folder

### resources
This folder contains the data, literature, log, etc.

### src
This folder contains the source code.

## Dependencies
- Python 3
- vaderSentiment
- nltk
- pandas
- numpy
- sklearn
- gensim

