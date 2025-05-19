#!/usr/bin/env python
# coding: utf-8

# In[172]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[173]:


df = pd.read_csv("train.csv")
testset = pd.read_csv("test.csv")


# In[174]:


print(df.head)
print(df.columns)
print(df.isnull().sum())
print(df.info())
print(df.describe())


# In[175]:


df.plot()
plt.show()


# In[176]:


df_encoded = df.copy()
for col in df_encoded.select_dtypes(include='object').columns:
    df_encoded[col] = df_encoded[col].astype('category').cat.codes

sns.heatmap(df_encoded.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Heatmap (with Encoded Categorical Features)')
plt.tight_layout()
plt.show()


# In[177]:


X = trainset.drop(['PassengerId', 'Survived','Name','Ticket','Cabin'],axis = 1)
y = trainset['Survived']
print(X.columns)


# In[178]:


# Encode 'Sex' column: male = 0, female = 1
X['Sex'] = X['Sex'].map({'male': 0, 'female': 1})

# Encode 'Embarked' column using one-hot encoding
X = pd.get_dummies(X, columns=['Embarked'], drop_first=True)


# In[179]:


X = X.fillna(X.mean())


# In[180]:


X = np.array(X).astype(float)
y = np.array(y).astype(float)
y = y.flatten()


# In[181]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train, y_test = train_test_split(X,y, test_size=0.2, train_size=0.8, random_state=42)


# In[182]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[183]:


print(type(X),X_train.shape)
print(type(y),y_train.shape)


# In[184]:


class LogReg:
    def __init__(self,lr=0.01,n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias= None
    def sigmoid(self,z):
        return 1/(1 + np.exp(-z))
    def fit(self,X,y):
        nsamp,nfeat = X.shape
        self.weights = np.zeros(nfeat)
        self.bias = 0
        for _ in range(self.n_iters):
            linearpred = np.dot(X,self.weights)+self.bias
            y_pred = self.sigmoid(linearpred)
            error = y_pred- y
            dw =  (1/nsamp) * np.dot(X.T,error)
            db = (1/nsamp)* np.sum(error)
            self.weights -= self.lr*dw
            self.bias -= self.lr*db
    def probab(self,X):
        linearpred = np.dot(X,self.weights)+self.bias
        y_pred = self.sigmoid(linearpred)
        return np.where(y_pred >= 0.5,1,0)


# In[185]:


model = LogReg(lr = 0.01,n_iters = 1000)
model.fit(X_train,y_train)


# In[186]:


predictions = model.probab(X_test)
accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy:.2f}")

