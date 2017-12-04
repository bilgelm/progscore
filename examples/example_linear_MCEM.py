
# coding: utf-8

# In[1]:

import progscore as ps
import numpy as np


# In[2]:

lt = ps.LinearTrajectory(1)


# In[8]:

type(np.ones(3).reshape(-1,1))


# In[7]:

type(np.array((1,0)).reshape(-1,1))


# In[9]:

np.matrix('1;0')


# In[13]:

np.matrix(np.ones(3).reshape(-1,1))


# In[16]:

np.matrix(np.eye(2))


# In[18]:

['a','b'] + [None]


# In[20]:

tmp = np.array((1,2)).reshape(-1,1)


# In[22]:

tmp.transpose() * tmp


# In[26]:

np.matrix([(1,2),(3,4)])


# In[32]:

np.concatenate((tmp,tmp),axis=1)


# In[34]:

np.sum(np.hstack((tmp,tmp)))


# In[ ]:



