#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

df = pd.read_csv('Advertising.csv')
print(df.head())


# In[2]:


import matplotlib.pyplot as plt

# Scatter plot for TV advertising
plt.scatter(df['TV'], df['Sales'])
plt.title('TV Advertisements')
plt.xlabel('TV Spending (in thousands)')
plt.ylabel('Sales (in thousands)')
plt.show()

# Similar plots for Radio and Newspaper
# ...


# In[3]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = df[['TV', 'Radio', 'Newspaper']]
X = scaler.fit_transform(X)


# In[4]:


from sklearn.model_selection import train_test_split

Y = df['Sales']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=50, test_size=0.25)


# In[5]:


from sklearn.linear_model import LinearRegression

clf = LinearRegression()
clf.fit(X_train, Y_train)


# In[6]:


from sklearn.metrics import mean_squared_error

Y_pred = clf.predict(X_test)
mse = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error:", mse)


# In[7]:


import statsmodels.api as sm

X2 = sm.add_constant(X)
est = sm.OLS(Y, X2)
est2 = est.fit()
print(est2.summary())


# In[ ]:




