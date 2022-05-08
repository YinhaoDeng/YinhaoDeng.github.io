---
layout:     post
title:      Authorship Attribution
subtitle:   a deep learning method
date:       2022-04-25
author:     Yinhao Deng, Zi Jin, Ruochen Mao
header-img: img/Authorship-attribution.jpeg
catalog: true
usemathjax: true
tags:
    - Statistical Machine Learning
    - Deep Learning
---

## 1 Introduction 
Authorship attribution deals with the article context similarity and author identification, the task is to identify the author of a given article. It has various applications including plagiarism detection and authorship deception identification. The goal of our task is to predict whether the target authors are the true authors of given documents from the test set. The training dataset contains 26108 published papers. The available information of each paper includes the publish year, publish venue, a set of keywords it contains and its authors. All features except year have been desensitized. We propose an embedding model solution that adapts embedding layers onto the Multilayer Perceptron model and outperforms our history-based baseline model.

## 2 Method
### 2.1 History-based Baseline Model: Naïve Approach
The baseline model is a lazy learner which is purely based on historical records, there is no learning involved. This model firstly converts features keywords, venue, year and co_author into one-hot encoding format. Secondly, the model sums up all the records to get the frequency of each feature for each author. During the inference phase, we measure the cosine similarity between the test sample of the target author with the author’s history record, and use the cosine similarity value as the predicted probability of whether the author is the true author of a given document. The baseline model obtained an overall AUC of 0.784 on the test set.

``` python
# import libraries
import json
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
```

```python
# load datasets
with open("../input/train.json", 'r') as train_f:
  train_data = json.load(train_f)

with open("../input/test.json", 'r') as test_f:
  test_data = json.load(test_f)

train_data_length = len(train_data)
X_train = np.zeros((train_data_length, 3293), dtype=int) # create an empty np array to store tshape = (26108, 3293)
author_history_table = np.zeros((2302, 3293), dtype=int) 
```

```python
train_data['0']  # one example of training data

# output
{'venue': '',
 'keywords': [64,
  1,
  322,
  134,
  136,
  396,
  270,
  144,
  476,
  481,
  165,
  39,
  361,
  43,
  177,
  308,
  310,
  118,
  187,
  127],
 'year': 2017,
 'author': [1605, 759]}```
```

Learn from historical records
```python
for i in range(train_data_length):  # iterate through train_data
    # local variables
    keywords = np.zeros((500), dtype=int)  # shape: (26108, 500)
    venue = np.zeros((471), dtype=int)  # shape: (26108, 471)  470 venues + '' one-hot
    year = np.zeros((20), dtype=int) # one-hot
    co_author = np.zeros((2302), dtype=int) # one-hot
    
    # keywords
    keywords_idx_list = train_data[str(i)]['keywords']
    for num in keywords_idx_list:
        keywords[num] = 1
    
    # venue
    venue_idx = train_data[str(i)]['venue']
    if venue_idx != '':
        venue[venue_idx] = 1  # assumption: only one venue!!!!!!
    
    # year
    year_ = train_data[str(i)]['year']  # extract the year int
    year[year_-2000] = 1

    # author
    author_idx_list = train_data[str(i)]['author']
    for author_idx in author_idx_list:
        author_idx_list_copy = author_idx_list
        co_author_idx_list = author_idx_list_copy.remove(author_idx) 
        
        if co_author_idx_list is True:
            for co_author_idx in co_author_idx_list:
                co_author[author_idx] = 1  # co_author one-hot encoding
    
        one_history_of_this_author = np.concatenate([keywords, venue, year, co_author])
        author_history_table[author_idx] += one_history_of_this_author  
```


author_history_table.shape:
```
(2302, 3293)
```

#### Test performance of baseline on the test set
```python
test_data_length = len(test_data)
y_pred = np.zeros((2000))

for i in range(test_data_length):  # iterate through train_data
    # local variables
    keywords = np.zeros((1, 500), dtype=int)  # shape: (26108, 500)
    venue = np.zeros((1, 471), dtype=int)  # shape: (26108, 471)  470 venues + '' one-hot
    year = np.zeros((1, 20), dtype=int) # one-hot
    co_author = np.zeros((1, 2302), dtype=int) # one-hot
    
    # keywords
    keywords_idx_list = test_data[str(i)]['keywords']
    for num in keywords_idx_list:
        keywords[0, num] = 1
    
    # venue
    venue_idx = test_data[str(i)]['venue']
    if venue_idx != '':
        venue[0, venue_idx] = 1  # assumption: only one venue!!!!!!
    
    # year
    year_ = test_data[str(i)]['year']  # extract the year int
    year[0, year_-2000] = 1

    # author
    co_author_idx_list = test_data[str(i)]['coauthor']
    
    for co_author_idx in co_author_idx_list:
        co_author[0, author_idx] = 1
    
    X_test_sample = np.concatenate([keywords, venue, year, co_author], axis=1)
    target_author_idx = test_data[str(i)]['target']
    
    from scipy import spatial
    cos_similarity = 1 - spatial.distance.cosine(author_history_table[target_author_idx], X_test_sample)
    y_pred[i] = cos_similarity
```


y_pred looks like this:

```
array([0.48852953, 0.2022783 , 0.36084058, ..., 0.24672081, 0.23077797,
       0.36128973])
```



## 2.2 Final Approach: Embedding model
### 2.2.1 Sampling
In this project, we have designed the authorship attribution task as a binary classification task. Therefore, both positive and negative samples are necessary for further development. For positive samples, we have extracted each author from the author list as the target author. For example, if a paper has two authors A and B, our training set would consist of two positive samples: {paper, author A} and {paper, author B}. This sampling method would allow us to train the model to learn directly the relationship between the paper and the target author. Whereas for negative samples, the target author is randomly selected from the negative pool. In this example, the negative pool would be a total of 2302 authors besides author A and B. Since random sampling has the limitation of not representing the full pattern of negative relationship, we have increased the number of negative samples to a ratio of 15:1, that is, for each positive sample, 15 negative samples are generated. The motivation of this sampling method is to allow the model to learn the true pattern between the negative and positive authors for each paper by providing as much information as possible, with the assumption that if a model is fully confident with true negative prediction, then consequently would have high accuracy in predicting true positives. 

### 2.2.2 Data preprocessing
There are four kinds of features in this project: year, venue, keywords and author. For feature venue and author, the original desensitized value is directly extracted and fed into an embedding layer (detail discussed in section 2.2.3). And for feature year, considering there are 20 different years in total, we have desensitised them into 0-19 instead, followed by inputting them into another embedding layer. In contrast, for feature keywords, Word2vec algorithm is used to pre-train and vectorise keywords before inputting them into the final model. The core idea of Word2vec is to represent a keyword as a multi-dimensional vector of continuous value, which could allow us to calculate how similar keywords (i.e. represented by vectors) are to each other. In this project, we have set the dimension of word2vec vectors to 50 since the total number of the keywords is relatively small with 500 vocabulary and a greater dimension tends to complex the model and is computationally expensive. In addition, in order to balance the input length of keywords for each paper, we have taken the average value of every keyword vector. Thus, the keywords feature for each paper is represented by an average vector of 50 dimensions.

### 2.2.3 Model 
In order to sufficiently detect authorship attribution using preprocessed data described in section 2.1, an embedding model is developed. Embedding is a technique used to learn and obtain a vector representation of a discrete variable. The advantage of using embedding is that the embedding vector has a relatively lower dimension compared to other traditional methods such as one-hot encoding, thus saving data memory storage and reducing computational complexity. Moreover, the vector representation learned using embedding is more informative and captures the interrelationship of similarity between each variable. Therefore, we have used 3 embedding layers to learn the vector representation of discrete features of year, venue and author respectively. In addition, the output size for embedding layers of each feature is specifically adjusted based on the difference between the original number of categories, ensuring a high-quality representation can be learned with obtaining maximal information about the corresponding feature and label, as well as avoiding high computational expense from extremely large embedding size. 

![model graph](https://github.com/YinhaoDeng/yinhao.github.io/blob/master/img/model-graph.svg?raw=true)
_Figure 1. The structure of the final approach model: word embedding model with MLP._

The overall structure of the embedding model is shown in figure 1. The model inputs year, venue, keywords and author separately. Among them, year, venue and author will be fed into an embedding layer, and the corresponding embedding vector representation will be produced. On the other hand, the input for keywords is a word2vec representation obtained from a pre-trained word2vec model (as described in section 2.2.2). Using a concatenate layer, the vector representations of 4 inputs are concatenated and fed into a Multilayer Perceptron (MLP). MLP is adapted as its advantage on the capability of learning non-linear relationships, which may be more suitable for the task. Since complex models (i.e. with deeper structure or greater number of nodes) tend to overfit with high variance, and contrastively over-simple models tend to underfit with high bias, we have carefully determined the number of hidden layers and the corresponding number of nodes in the way that maximising the generalised performance and reducing computational expense. Moreover, normalisation and dropout layers are used to stabilise the model and prevent overfitting. Finally, the performance of our final approach using the embedding model achieved a validation score of AUC 0.87. And the training AUC curve and the loss curve are shown in Figure 2&3 below. From both curves, we can see that the validation score and loss converge after epoch 15, whereas the training score/loss is continuously increasing/decreasing, which indicates that the model tends to slightly overfit. One possible solution to overfitting would be increasing the sample size of both positive and negative samples. 


![training curve](https://github.com/YinhaoDeng/yinhao.github.io/blob/master/img/training-curve.png?raw=true)
_Figure 2. Train/val AUC over epoc_	  

![training curve](https://github.com/YinhaoDeng/yinhao.github.io/blob/master/img/train-val-loss.png?raw=true)
_Figure 3. Train/val loss over epoch_
## 3 Discussion
### 3.1 Comparison with other approach
We have also experimented various approaches and discovered some interesting findings. One experimental approach is developing a simple MLP model (i.e. with similar structure as the second half of the embedding model) that inputs features year, venue and author as one-hot representation vectors, rather than using embedding layers to encode those features. Unfortunately, this MLP model only achieved a test AUC score of 0.79, which is a lower performance compared to our final embedding approach. One possible reason for this lower performance would be that one-hot encoding is less informative with the assumption of independence between each entity. Whereas embedding method has the ability to capture and represent the interrelationship such as similarities between each entity and thus providing additional information for the model to learn the relationship between authors and papers, resulting in higher performance. Moreover, the one-hot encoding is extremely sparse and would consume more memory resources during training. With the limitation of memory storage (16GB on Kaggle kernel), using embedding methods with inputting original desensitized variables would save more memory space and allow us to generate more samples for the model to learn and generalise better. Therefore, we can conclude that our final approach, the embedding model, has the advantage of being more informative and having lower memory cost, which leads to better performance. 

### 3.2 Negative sampling
Our final approach would also suffer from the disadvantage of the negative sampling strategy. As discussed in section 2.2.1, in order to represent a comprehensive pattern of negative relationships, for each positive sample, we generated 15 negative samples. We have also experimented other sampling ratios such as 1:5 and 1:10, but the performance of reducing ratio is much lower and less generalised. One possible reason might be that the true pattern of negative relationships could not be represented well with reduced ratios. However, this 1:15 strategy would lead to an imbalance between the positive and negative samples, causing the model to learn more about negative relationships than positive relationships, as a result, favouring negative prediction, thereby increasing the false negative rate. For future work, a better negative sampling strategy could be used to minimise this disadvantage. For example, instead of increasing quantity, negative samples with higher quality could be generated by computing the similarity between each paper, that is, selecting the less similar author as negative samples. 
