#!/usr/bin/env python
# coding: utf-8

# # Task1: Prediction using Supervised Machine Learning
# ## By Vikram Jeet Singh

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


link='http://bit.ly/w-data'
data=pd.read_csv(link)
data


# In[3]:


a=data['Hours']
b=data['Scores']
plt.scatter(a,b)
plt.xlabel('Study Hours')
plt.ylabel('Marks Scored')
plt.show()


# ## By observation we can see that there is a direct relationship between study hours and marks scored.

# In[18]:


#Now lets prepare the data
x=data['Hours'].to_numpy()
x=x.reshape(-1,1)
y=data["Scores"].to_numpy()
y=y.reshape(-1,1)
x,y


# In[38]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)


# In[39]:


from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)
print('Trained the model')


# In[40]:


line=model.coef_*x + model.intercept_
plt.plot(x,line)
plt.scatter(x,y)
plt.xlabel('Study Hours')
plt.ylabel('Marks Scored')
plt.show()


# In[41]:


print(x_test)
y_pred=model.predict(x_test)


# In[60]:


y_pred


# In[74]:


y_test.flatten()


# In[76]:


pd.DataFrame(data={'Predicted Value':y_pred.flatten(),'Actual Value':y_test.flatten()})


# ### Now as we can see that our model was quite successful in predicting the values, let's calculate the error that how much do the predicted values differ from actual values.

# In[79]:


#Mean Absolute Error
from sklearn.metrics import mean_absolute_error
mae=mean_absolute_error(y_pred,y_test)
print("Hence the mean absolute error is ",mae)


# # Now let us do the main task of prediction of percentage for a student who studies 9.25hrs/day

# In[84]:


target=[[9.25]]
prediction=model.predict(target)
prediction


# # Hence, the student who studies for 9.25 hrs/day willl score 93.69173249% in the end of the year approximately...

# In[ ]:




