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


# In[4]:


X = df.data
y= df.target
X = pd.DataFrame(df.data)
X = X.fillna(X.mean())
y=y.flatten()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33, random_state=42)


# In[11]:


#K-Nearest Neighbour with euclidean metric
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
accuracies = []
ks = range(1,21)
for k in ks:
    knn = KNeighborsClassifier(n_neighbors=k,metric='euclidean')
    knn.fit(X_train,y_train)
    y_pred_knn = knn.predict(X_test)
    acc = accuracy_score(y_test,y_pred_knn)
    accuracies.append(acc)
print("KNN Results:")
print(confusion_matrix(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))


# In[13]:


#Decision Tree
from sklearn.tree import DecisionTreeClassifier, plot_tree
clf = DecisionTreeClassifier(max_depth = 3, random_state = 42)
clf.fit(X,y)
y_pred_dt = clf.predict(X_test)
plt.figure(figsize = (15,8))
plot_tree(clf,feature_names = df.feature_names,class_names = df.target_names,filled = True)
plt.show()
print("Decision Tree Results:")
print(confusion_matrix(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))


# In[14]:


#Random Forest 
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=42, n_estimators = 100)
rf.fit(X_train,y_train)
y_pred_rf = rf.predict(X_test)
print("Random Forest Results:")
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))


# In[17]:


#compare these three algorithms using heatmap 
#1.Random Forest
import matplotlib.pyplot as plt
import seaborn as sns
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Blues')
plt.title("Random Forest Confusion Matrix")
plt.show()

#2. Decision Tree
import matplotlib.pyplot as plt
import seaborn as sns
sns.heatmap(confusion_matrix(y_test, y_pred_dt), annot=True, fmt='d', cmap='Blues')
plt.title("Decision Tree Confusion Matrix")
plt.show()

#3. KNN 
import matplotlib.pyplot as plt
import seaborn as sns
sns.heatmap(confusion_matrix(y_test, y_pred_knn), annot=True, fmt='d', cmap='Blues')
plt.title("KNN Confusion Matrix")
plt.show()


# In[20]:


import pandas as pd

# Creating a DataFrame with metric values
data = {
    'Model': ['KNearestNeighbour', 'Decision Tree', 'Random Forest'],
    'Accuracy': [0.97, 0.93, 0.97],
    'Precision': [0.97, 0.93, 0.97],
    'Recall': [0.97, 0.93, 0.97],
    'F1-Score': [0.97, 0.93, 0.97]
}

df_results = pd.DataFrame(data)
print(df_results)

