# Credit Card Fraud Detection

## Table of contents
- [Introduction](#introduction)
- [Data Overview](#data-overview)
- [Model Implementation](#model-implementation)
  - [Autoencoder](#autoencoder)
  - [Gaussian Mixture Model](#gaussian-mixture-model)
  - [Random Forest](#random-forest)
  - [Gradient Boosting](#gradient-boosting)
- [Conclusion](#conclusion)

## Introduction

In this project, I implemented 4 machine learning models for anomaly detection on Kaggle's [credit card fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data) dataset.

They determine if a transaction is fraudulent or regular, i.e. a binary classification, from a set of 30 transaction features. However, the dataset is naturally highly unbalanced, with the vast majority (99%+) of data being non-fraudulent.

The models implemented are Gaussian Mixture Model, Random Forest, Gradient Boosting, and Autoencoder.

## Data Overview

The dataset contains 30 parameters of the transaction, and a binary result indicating its validity, with 0 being regular, and 1 being fraudulent.

Of the 30 parameters in the input, 2 are listed as being time in seconds and the amount of money, while the other 28 are principal components taken from PCA of some other confidential data.

Due to the data being highly imbalanced, I followed Kaggle's recommendation to show the area under the [precision-recall curve](https://medium.com/@douglaspsteen/precision-recall-curves-d32e5b290248) to gauge the quality of models trained on the dataset. It helps to also analyze the precision-recall tradeoff, not just the behavior on the dataset.

Additionally, the data was split into a training set and a test set, with the training set being 80% of the original data.

## Model Implementation

### Autoencoder

An autoencoder is a neural network that has layers that narrow down into a bottleneck with fewer neurons, and then widen back up with an output layer as big as input. Generally, it is used for denoising, as the bottleneck makes the neural network capture the most essential components, to then reconstruct the data.

It is applicable to anomaly detection as well, where it is trained to reconstruct the normal data, and then flags anomalies if a particular datapoint has a very high reconstruction error. However, as I demonstrate in the `Fraud_Detection_Autoencoder.ipynb` notebook, that technique was rather unwieldy for this dataset.

I used Mean-Squared-Error, and showed on a plot that the reconstruction error was small for normal points, and big for fraudulent points. But setting a threshold for error wasn't enough to get an accuracy better than a coin flip.

So I had to deploy scikit-learn's logistic regressor to take reconstruction error and then output normal/fraudulent, also while using oversampled data to improve the dataset balance, turning it into a hybrid model. Even that has so far achieved the worst precision-recall curve area score. 

### Gaussian Mixture Model

The Gaussian Mixture Model (GMM) works similarly to clustering, however, it applies soft clustering in which points are assigned with various probabilities to various groups. Specifically, each cluster is assumed to be multivariate Gaussian, which will contain a quadratic form from its covariance matrix that will ultimately form an ellipse. 

Bottom line is, each cluster is modelled as a multivariate Gaussian distribution, the contours of which are learned from the dataset.

I deployed a GMM model using scikit-learn, in which I had to slowly tune the number of Gaussian components, which was optimally sitting at 9.

On the training dataset, the precision-recall curve appeared like so, with a pretty good area!

![image](https://github.com/user-attachments/assets/7ebcacbb-b036-42c1-8863-1c2d6076d737)

Besides the scikit-learn implementation, I also tried a from-scratch JAX implementation, which still struggles with precision-recall but is much faster.

### Random Forest

Random Forest combines multiple decision trees, each trained on a subset of the data and using a subset of the features. All trees vote to decide the model's outcome. 

After deploying it with scikit-learn as well, the precision-recall curve improved! The graph below is the curve on the _test_ dataset!

![image](https://github.com/user-attachments/assets/e70011cf-b745-4e97-b6e4-118317336f32)

### Gradient Boosting

The gradient boosting algorithm is an ensemble method that builds up trees to iteratively minimize loss using gradient descent. 

Applied using XGBoost with a logistic objective function using 10 threads to speed up each tree's construction, it completed running in 15 seconds.

Moreover, it was also the most accurate. It achieved the best possible AUC precision-recall score. For the training set, it was a perfect 1.0:

![image](https://github.com/user-attachments/assets/21a6ce48-04c3-4dc6-9203-97c55f8a8c65)

For the test set, it was nearly 0.9:

![image](https://github.com/user-attachments/assets/f65b98c5-fcb7-4409-b2cd-73dba00ad1da)

## Conclusion

XGBoost's gradient boosting was not only the fastest of the models, but also the most accurate.

Let's see what comes up from the JAX Gaussian Mixture Model. I believe it shows promise!
