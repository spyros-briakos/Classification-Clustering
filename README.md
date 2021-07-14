# Classification-Clustering
The purpose of this project is to be familiarized with the basic steps of the process followed for the application of data mining techniques, namely: collection, pre-processing/cleaning, conversion, application of data mining techniques and evaluation. The implementation will be done in the Python programming language using the tools/libraries such us jupyter notebook, pandas, gensim and SciKit Learn.
Check it here: [![Click here to open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/spympr/Classification-Clustering/blob/main/Data_Mining_Project_2.ipynb)

## Description
This work relates to the categorization of text data by news articles. The dataset we have to deal with contains news files in .txt format which belong to 5 categories (business, entertainment, politics, sport, tech). We have been asked to create the following .tsv files (tab separated files), ie files in which the fields of the records are separated by the character '\t' (tab). In each .txt file the first line is the title of their article.  


**1.** **train_set.csv** (will be 80% of the total data points): This file will be used to train your algorithms and contains the following fields:  
**a. Id**: A unique number for article  
**b. Title**: The title of article  
**c. Content**: The content of the articled.  
**Category**: The category in which the article belongs.


**2. test_set.csv** (20% of data points): This file will be used to make predictions for new data. Contains all fields of the training file except the ‘Category’ field. We have been asked to estimate this field using categorization algorithms. 

## 1. Creating WordCloud 

At this point we have been asked to create a WordCloud for the five article categories. To create a WordCloud we used the text from all
articles in each category.  

## 2.Classification Implementation  
In this query we tried the following Classification methods:  
● Support Vector Machines (SVM, experiment with kernel parameters (rbf, linear ), c and gamma. The parameters can also be selected with GridSearchCV)  
● Random Forests  
● Naive Bayes  
● K-Nearest Neighbor (We didn't use any implementation of the algorithm provided by the library. In the implementation of K-Nearest Neighbor we have used the Majority Voting for the choice of the final label.)  

The categorization have been done in the following different representations of the texts: In the corresponding document-words table that will result from BoW representation of the texts and separately in the tf-idf transformation of the counts.  
We also evaluated and recorded the performance of each method using 10-fold Cross Validation using the following metrics:  
● Precision / Recall / F-Measure  
● Accuracy  
● ROC plot

## 3.Beat the Benchmark (bonus)  
Finally we have been asked to experiment with any Classification method we want, doing any pre-processing of the data we want with the aim to beat our performance in the previous query as much as possible.

## 4.Clustering Implementation  
In this query we have been asked to implement clustering in the various text files. The number of clusters for each query will be 5. The clustering will be used K-Means clustering algorithm. The distance function to be used is Cosine Similarity. K-Means will be applied to the data
training set. The clustering should be implemented without using the Category variable. The clustering should be done in the following different representations of the texts:  
● In the corresponding document-words table that will result from BoW representation of the texts (both in simple counts, and separately in the tf-idf transformation of the counts)  
● In the corresponding document-embeddings table that will resulting from using pre-trained embeddings (one of the word2vec, glove, fast-text). To be able to view points in 2d space, use a compression method of Principal Component Analysis (PCA), Singular Value Decomposition (SVD) or Independent Component Analysis (ICA) (if you use all 3 you get a bonus). You will apply compression method to both of the previously mentioned representations.
