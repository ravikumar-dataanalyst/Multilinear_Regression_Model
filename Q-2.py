#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns 
import matplotlib.pyplot as plt

# Metrics packages
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn import datasets, linear_model, metrics


# In[2]:


# Load your dataset (replace 'your_dataset.csv' with your data file)
# 
data = pd.read_csv('weatherHistory.csv')
columns_to_drop = ['Loud Cover', 'Formatted Date']
data = data.drop(columns=columns_to_drop)
print(data.head())


# In[3]:


# Define your independent variables (features) and dependent variable (target)
X = data[['Humidity', 'Wind Speed (km/h)', 'Visibility (km)']]  # Replace with your feature columns
y = data['Temperature (C)'] 


# In[4]:


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[5]:


# Create a linear regression model
model = LinearRegression()


# In[6]:


# Train the model on the training data
model.fit(X_train, y_train)


# In[7]:


# Make predictions on the test data
y_pred = model.predict(X_test)


# In[8]:


# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# In[9]:


print(f"Mean Squared Error: {mse}")


# In[10]:


print(f"R-squared (R2) Score: {r2}")


# In[14]:


# EDA: TEMP distributed normally with some exceptions
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.distplot(data['Temperature (C)'], bins=30)
plt.show()


# In[ ]:




