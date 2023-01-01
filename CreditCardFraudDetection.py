#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[6]:


# loading the dataset to a Pandas DataFrame
credit_card_data = pd.read_csv("C:\\Users\\YAMINI KHANDURI\\OneDrive\\Desktop\\Yamini Khanduri.csv")


# In[7]:


# first 5 rows of the dataset
credit_card_data.head()


# In[8]:


credit_card_data.tail()


# In[9]:


# dataset informations
credit_card_data.info()


# In[10]:


# distribution of legit transactions & fraudulent transactions
credit_card_data['Class'].value_counts()


# In[11]:


# separating the data for analysis
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]


# In[12]:


print(legit.shape)
print(fraud.shape)


# In[13]:


# compare the values for both transactions
credit_card_data.groupby('Class').mean()


# In[14]:


legit_sample = legit.sample(n=492)


# In[15]:


new_dataset = pd.concat([legit_sample, fraud], axis=0)


# In[16]:


new_dataset.head()


# In[17]:


new_dataset.tail()


# In[18]:


new_dataset['Class'].value_counts()


# In[19]:


new_dataset.groupby('Class').mean()


# In[20]:


X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']


# In[21]:


print(X)


# In[22]:


print(Y)


# In[24]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)


# In[25]:


print(X.shape, X_train.shape, X_test.shape)


# # LOGISTIC REGRESSION

# In[29]:


model = LogisticRegression()


# In[30]:


# training the Logistic Regression Model with Training Data
model.fit(X_train, Y_train)


# In[31]:


# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[32]:


print('Accuracy on Training data : ', training_data_accuracy)


# In[33]:


# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[34]:


print('Accuracy score on Test Data : ', test_data_accuracy)

