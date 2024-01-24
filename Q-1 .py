#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.model_selection import train_test_split

# Data manipulations
import numpy as np
import pandas as pd

#Visualizations
import seaborn as sns 
import matplotlib.pyplot as plt

# Metrics packages
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn import datasets, linear_model, metrics


# In[2]:


df = pd.read_csv("Updated Cars Data.CSV") # Talk about other parameters sep="\s+",skiprows=22, header=None)
print(df.head(2))


# In[7]:


# create dummy variables
df = pd.get_dummies(df)
print(df.head(2))

# We select numerical features

X = df.loc[:, df.columns != 'CO2']
y = df['CO2']


# In[4]:


# EDA: CO2 distributed normally with some exceptions
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.distplot(df['CO2'], bins=30)
plt.show()


# In[8]:


reg = linear_model.LinearRegression()
reg.fit(X, y)


# In[9]:


# regression coefficients
print('Coefficients: ', reg.coef_)

# r2_score score
print('r2_score: {}'.format(r2_score(y, reg.predict(X))))

rmse = (np.sqrt(mean_squared_error(y, reg.predict(X))))
print('RMSE is {}'.format(rmse))

# setting plot style
plt.style.use('fivethirtyeight')

# plotting residual errors in training data, y^-y
plt.scatter(reg.predict(X),
			reg.predict(X) - y,
			color="green", s=10,
			label='Train data')

# plotting line for zero residual error
plt.hlines(y=0, xmin=0, xmax=150, linewidth=2)

# plotting legend
plt.legend(loc='upper right')

# plot title
plt.title("Residual errors")

# method call for showing the plot
plt.show()


# In[ ]:




