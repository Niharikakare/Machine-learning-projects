#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import pickle


# In[9]:


df= pd.read_csv(r"C:\Users\mouni\Downloads\large_house_price_dataset.csv")
df


# In[10]:


df.isnull()


# In[11]:


df.count()


# In[12]:


df.info()


# In[13]:


df.shape


# In[14]:


df.describe()


# In[15]:


df.isnull().sum()


# In[16]:


df.duplicated().sum()


# In[17]:


plt.scatter(df["Area"],df["Price"])
plt.xlabel("Area (sqft)")
plt.ylabel("Price")
plt.show()


# In[18]:


X=df[["Area","Bedrooms","Bathrooms","Floors","Parking","Age"]]
Y=df["Price"]
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20,random_state=40)


# In[19]:


lr=LinearRegression()
lr.fit(X_train, Y_train)


# In[20]:


Y_train_pred = lr.predict(X_train)
Y_test_pred = lr.predict(X_test)


# In[21]:


print("Train R2:" ,r2_score(Y_train,Y_train_pred))
print("Test R2:", r2_score(Y_test,Y_test_pred))


# In[22]:


mae= mean_absolute_error(Y_test,Y_test_pred)
mse= mean_squared_error(Y_test,Y_test_pred)
rmse = np.sqrt(mse)

print("MAE :" ,mae)
print("MSE :" ,mse)
print("RMSE :" ,rmse)


# In[23]:


ridge = Ridge(alpha=1.0)
ridge.fit(X_train,Y_train)
ridge_pred = ridge.predict(X_test)
print("Ridge R2:" , r2_score(Y_test,ridge_pred))


# In[24]:


X_train.shape


# In[25]:


X_test.shape


# In[26]:


lasso = Lasso(alpha=0.1)
lasso.fit(X_train,Y_train)
lasso_pred = ridge.predict(X_test)
print("Lasso R2:" , r2_score(Y_test,lasso_pred))


# In[27]:


models = {
       "Linear": r2_score(Y_test,Y_test_pred),
       "Ridge": r2_score(Y_test,ridge_pred),
       "Lasso": r2_score(Y_test,lasso_pred),
}
models


# In[28]:


with open("model.pkl" , "wb")as file:
    pickle.dump(ridge, file)


# In[ ]:




