#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# In[2]:


from sklearn.datasets import load_iris
df = load_iris()


# In[3]:


df.data


# In[4]:


df.target


# In[5]:


X = df.data
y= df.target


# In[6]:


print(X.shape)
print(y.shape)


# In[7]:


X = pd.DataFrame(df.data)
X = X.fillna(X.mean())
y=y.flatten()


# In[8]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33, random_state=42)


# In[20]:


from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
accuracies = []
ks = range(1,21)
for k in ks:
    knn = KNeighborsClassifier(n_neighbors=k,metric='euclidean')
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test,y_pred)
    accuracies.append(acc)


# In[21]:


plt.figure()
plt.plot(ks, accuracies, marker='o')
plt.title('KNN: Accuracy vs k')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.xticks(ks)
plt.grid(True)
plt.show()


# In[22]:


#PCA
#standardizing the scaler
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[23]:


#apply pca
from sklearn.decomposition import PCA

pca = PCA(n_components=2)  # Reduce to 2 dimensions
X_pca = pca.fit_transform(X_scaled)


# In[24]:


#visualize it
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k', s=70)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Iris Dataset')
plt.grid(True)
plt.legend(*scatter.legend_elements(), title="Classes")
plt.show()


# In[25]:


print("Explained variance ratio:", pca.explained_variance_ratio_)

