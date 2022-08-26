#!/usr/bin/env python
# coding: utf-8

# In[56]:


import numpy as np
import pandas as pd


# In[57]:


ecg = pd.read_csv('ECG200_TRAIN.csv' , delimiter = '  ',header =None)


# In[58]:


ecg


# In[ ]:





# In[93]:


labels=['feature '+str(x) for x in range(ecg.shape[1])]
labels


# In[60]:


labels[0]='class'


# In[61]:


labels


# In[62]:


ecg.columns = labels


# In[63]:


ecg.head()


# In[64]:


labels = ecg.iloc[0:100, :1]


# In[65]:


labels.value_counts()


# In[66]:


ecg1 = ecg[ecg['class'] == 1.0]
ecg2 = ecg[ecg['class'] == -1.0]


# In[67]:


ecg1.shape


# In[68]:


ecg2.shape


# In[69]:


rem1 = int(69/4)
rem2 = int(31/4)
print("ans 1 :", rem1)
print("ans 2 :", rem2)


# In[70]:


df1 = pd.concat([ecg1[:rem1], ecg2[:rem2]])

df2 =  pd.concat([ecg1[rem1 : 2*rem1], ecg2[rem2 : 2*rem2]])

df3 =  pd.concat([ecg1[2*rem1 : 3*rem1], ecg2[2*rem2 : 3*rem2]])

df4 =  pd.concat([ecg1[3*rem1 :], ecg2[3*rem2 :]])


# In[71]:


(df1.shape, df2.shape, df3.shape, df4.shape)


# In[ ]:





# In[72]:


df1['class'].value_counts()


# In[73]:


df2['class'].value_counts()


# In[74]:


df3['class'].value_counts()


# In[75]:


from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report 


# In[76]:


df1 = shuffle(df1)
df2 = shuffle(df2)
df3 = shuffle(df3)
df4 = shuffle(df4)


# In[ ]:





# In[77]:


df4['class'].value_counts()


# In[78]:


x = pd.concat([df2, df3, df4], axis = 0)
y = df1
y.head()


# In[79]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

acc1 = []
acc2 = []

X_train = x.drop('class',axis=1)
Y_train = x["class"]
X_test = y.drop('class',axis=1)
Y_test = y["class"]

c1 = RandomForestClassifier(max_depth=None, random_state=0)
c1.fit(X_train, Y_train)
p = c1.predict(X_test)
print ("Accuracy : " , accuracy_score(Y_test,p)*100)  
print("Report : \n", classification_report(Y_test, p))
print("F1 Score : ",f1_score(Y_test, p, average='macro')*100)

acc1.append(accuracy_score(Y_test,p)*100)


# In[80]:


x = pd.concat([df1, df3, df4], axis = 0)
y = df2

X_train = x.drop('class',axis=1)
Y_train = x["class"]
X_test = y.drop('class',axis=1)
Y_test = y["class"]

c1 = RandomForestClassifier(max_depth=None, random_state=0)
c1.fit(X_train, Y_train)
p = c1.predict(X_test)
print ("Accuracy : " , accuracy_score(Y_test,p)*100)  
print("Report : \n", classification_report(Y_test, p))
print("F1 Score : ",f1_score(Y_test, p, average='macro')*100)

acc1.append(accuracy_score(Y_test,p)*100)


# In[81]:


x = pd.concat([df1, df2, df4], axis = 0)
y = df3


X_train = x.drop('class',axis=1)
Y_train = x["class"]
X_test = y.drop('class',axis=1)
Y_test = y["class"]

c1 = RandomForestClassifier(max_depth=None, random_state=0)
c1.fit(X_train, Y_train)
p = c1.predict(X_test)
print ("Accuracy : " , accuracy_score(Y_test,p)*100)  
print("Report : \n", classification_report(Y_test, p))
print("F1 Score : ",f1_score(Y_test, p, average='macro')*100)

acc1.append(accuracy_score(Y_test,p)*100)


# In[82]:


x = pd.concat([df1, df2, df3], axis = 0)
y = df4

X_train = x.drop('class',axis=1)
Y_train = x["class"]
X_test = y.drop('class',axis=1)
Y_test = y["class"]

c1 = RandomForestClassifier(max_depth=None, random_state=0)
c1.fit(X_train, Y_train)
p = c1.predict(X_test)
print ("Accuracy : " , accuracy_score(Y_test,p)*100)  
print("Report : \n", classification_report(Y_test, p))
print("F1 Score : ",f1_score(Y_test, p, average='macro')*100)

acc1.append(accuracy_score(Y_test,p)*100)


# # Using Sklearn

# In[83]:


df = pd.read_csv('ECG200_TRAIN.csv' , delimiter = '  ',header =None)


# In[84]:


from sklearn.model_selection import train_test_split

x = ecg.drop('class',axis=1)
y = ecg['class']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 40 )

c4 = RandomForestClassifier(max_depth=None, random_state=0)
c4.fit(x_train, y_train)
p = c4.predict(x_test)
print ("Accuracy : " , accuracy_score(y_test,p)*100)  
print("Report : \n", classification_report(y_test, p))
print("F1 Score : ",f1_score(y_test, p, average='macro')*100)


# # Kfold Cross Validation

# In[87]:


from sklearn.model_selection import KFold
kfol = KFold(n_splits = 4,shuffle = True)
for k, f in kfol.split(x): 
    X_train = x.iloc[k]
    X_test  = x.iloc[f]
    y_train = y.iloc[k]
    y_test  = y.iloc[f]    
    cl = RandomForestClassifier()
    cl = cl.fit(X_train, y_train)
    p = cl.predict(X_test)
    cl.score(X_test,y_test)

    print ("Accuracy : " , accuracy_score(y_test,p)*100)
    acc2.append(accuracy_score(y_test,p)*100)


# In[88]:


print ("Accuracies without sklearn :")
acc1


# In[89]:


print ("Accuracies with sklearn :")
acc2


# In[91]:


print("The average accuracy Using Numpy: ", (sum(acc1)/4))
print("The average accuracy Using Sklearn: ", (sum(acc2)/4))


# #  In Conclusion the average accuracy result using Numpy is better than that using Sklearn

# In[ ]:




