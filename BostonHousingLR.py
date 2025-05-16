#!/usr/bin/env python
# coding: utf-8

# In[313]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[315]:


coln = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
    "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"
]
#delimwhitespace batata hai ki columns spaces se divide kar diya gaya hai
#header none set krne se data ko pata chlta h ki first row ko column name jaise treat na kare
df = pd.read_csv("housing.csv",delim_whitespace = True,header = None)
df.columns = coln
print(df.columns)


# In[310]:


df.plot()
plt.show()


# In[311]:


df.plot.box()
plt.show


# In[345]:


#axis = 1 set krne se column wise operate hua, jisse ki pura column drop kiya 
#agr axis = 0 set krte, to row wise operate hota, or MEDV naam ke row ko drop krne ki kosis krta
#https://www.kaggle.com/code/prasadperera/the-boston-housing-dataset - used this data preprocessing
#got to know, these variables works the best, he has done excellent job, check it out
#from my knowledge of graphs i got to know that, ZN,AGE,CHAS doesnt has that much effect in prediction
column_sels = ['LSTAT', 'INDUS', 'NOX', 'PTRATIO', 'RM', 'TAX', 'DIS']
x = df.loc[:,column_sels]
y = df["MEDV"]


# In[355]:


import seaborn as sns
import matplotlib.pyplot as plt

# Selected columns
columns_of_interest = ['LSTAT', 'INDUS', 'NOX', 'PTRATIO', 'RM', 'TAX', 'DIS', 'MEDV']
subset = df[columns_of_interest]

# Calculate correlation with MEDV
medv_corr = subset.corr()['MEDV'].drop('MEDV')

# Plot barplot
plt.figure(figsize=(10, 6))
sns.barplot(x=medv_corr.values, y=medv_corr.index, palette='coolwarm')
plt.title("Correlation of Selected Features with MEDV")
plt.xlabel("Correlation Coefficient")
plt.ylabel("Feature")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()


# In[346]:


# feature normalization
# isse sare features ko same scale pe le aayege jisse LR use kr ke predict krne me easy ho
X_normalized = (X - X.min()) / (X.max() - X.min())
Xnp = X_normalized.values


# In[347]:


print(np.max(Xnp, axis=0))
print(np.min(Xnp, axis=0))
print(np.mean(Xnp, axis=0))
print(np.std(Xnp, axis=0))


# In[348]:


print(Xnp)


# In[349]:


import numpy as np
class LinearRegression:
    def __init__(self,lr=1e-5,n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weight = None
        self.bias = None
        
    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weight = np.random.rand(num_features)
        self.bias = 0

        for i in range(self.n_iters):
            y_pred = np.dot(X, self.weight) + self.bias
            error = y_pred - y

            dw = (1 / num_samples) * np.dot(X.T, error)
            db = (1 / num_samples) * np.sum(error)

            # Debug print every 100 iterations or when NaN appears
            if i % 100 == 0 or np.isnan(dw).any() or np.isnan(db):
                print(f"Iteration {i}")
                print(f"dw: {dw}")
                print(f"db: {db}")
                print(f"weight: {self.weight}")
                print(f"bias: {self.bias}")
                print(f"y_pred[:5]: {y_pred[:5]}")
                print(f"error[:5]: {error[:5]}")

            # Detect crash
            if np.isnan(dw).any() or np.isnan(db):
                print("NaN occurred! Breaking loop.")
                break

            self.weight -= self.lr * dw
            self.bias -= self.lr * db

        return self

    def predict(self,X):
        return np.dot(X,self.weight)+self.bias


# In[350]:


#dataframe ko ab array me convert kro taki values easy ho calculate krne me
Xnp = X.values
ynp = y.values


# In[351]:


# Check shapes
print(Xnp.shape, ynp.shape)
ynp = ynp.ravel()

# Check NaNs
print(np.isnan(Xnp).any(), np.isnan(ynp).any())


# In[352]:


#initialization and training of model
model = LinearRegression(lr=0.001, n_iters=1000)
model.fit(Xnp, ynp)


# In[353]:


y_pred = model.predict(Xnp)


# In[354]:


from sklearn.metrics import mean_squared_error, r2_score

print("Mean Squared Error:", mean_squared_error(ynp, y_pred))
print("R² Score:", r2_score(ynp, y_pred))

