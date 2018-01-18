
# coding: utf-8

# In[15]:

import progscore as ps
import numpy as np
import pandas as pd

import scipy as sp
from scipy import linalg as sp_linalg


# In[2]:

maxNumVisits = 7
numBiomarkers = 8
numSubjects = 100
numVisitsPerSubject = np.random.choice(maxNumVisits-1, numSubjects)+1

numSubjVisits = np.sum(numVisitsPerSubject)

# Ground truth subject-specific variables
ubar_truth = np.array((0.05, -3.80)).reshape(-1,1)
V_truth = np.matrix([(0.0055, -0.4),(-0.4, 30)])
u_truth = np.random.multivariate_normal(ubar_truth.flatten(), V_truth, numSubjects)
alpha_truth = u_truth[:,0]
beta_truth = u_truth[:,1]

# Ground truth trajectory parameters
a_truth = np.random.rayleigh(0.09,numBiomarkers)-0.05
b_truth = np.random.normal(1.1,0.15,numBiomarkers)

# Generate age at baseline
age_baseline = np.random.uniform(56,93,numSubjects)

# Subject IDs
subject = np.repeat(np.arange(numSubjects),numVisitsPerSubject)
# All subjects are controls
dx = np.ones_like(subject)

# Generate visit numbers and ages at follow-up visits
visit = np.zeros_like(subject)
age = np.zeros_like(subject, dtype=float)


# In[3]:

for i in range(numSubjects):
    idx = subject==i
    vi = numVisitsPerSubject[i]
    visit[idx] = np.arange(vi)
    intervals = np.random.uniform(1,3,vi-1)
    age[idx] = np.cumsum(np.insert(intervals,0,age_baseline[i]))


# In[4]:

# Compute ground truth PS values
ps_truth = np.repeat(alpha_truth,numVisitsPerSubject) * age +            np.repeat(beta_truth,numVisitsPerSubject)

mu = np.mean(ps_truth)
sdev = np.std(ps_truth)
alpha_truth = alpha_truth / sdev
beta_truth = (beta_truth - mu) / sdev
ubar_truth[1] -= mu
ubar_truth /= sdev
V_truth /= sdev**2

ps_truth = np.repeat(alpha_truth,numVisitsPerSubject) * age +            np.repeat(beta_truth,numVisitsPerSubject)


# In[5]:

trajectory_truth = ps.LinearTrajectory(numBiomarkers)
p = np.vstack((a_truth,b_truth)).T
trajectory_truth.setParams(p)


# In[6]:

y_truth = trajectory_truth.predict(ps_truth)


# In[7]:

lmbda_truth = np.random.rayleigh(0.04,numBiomarkers)+0.08

C_truth = np.eye(numBiomarkers)
R_truth = np.diag(np.square(lmbda_truth))


# In[8]:

y = y_truth + np.random.multivariate_normal(np.zeros(numBiomarkers), R_truth, numSubjVisits)


# In[9]:

data = pd.DataFrame({'subjectIDps': subject, 
                     'age': age,
                     'dx': dx})

model_truth = ps.LinearPSModel(data, y=y)


# In[10]:

model_truth.trajectory = trajectory_truth
model_truth.lmbda = lmbda_truth.reshape(-1,1)
model_truth.subjectParameters['ubar'] = ubar_truth.reshape(-1,1)
model_truth.subjectParameters['V'] = V_truth
model_truth.subjectVariables['alpha'] = alpha_truth.reshape(-1,1)
model_truth.subjectVariables['beta'] = beta_truth.reshape(-1,1)
model_truth.C = C_truth
model_truth.R = R_truth


# In[12]:

# let's look at first individual
i = 0
idx = subject==i
agei = age[idx]
yi = y[idx,:]

print(agei)
print(yi)


# In[23]:

# 
a = model_truth.trajectory.params[:,0].reshape(-1,1)
b = model_truth.trajectory.params[:,1].reshape(-1,1)
ubar = model_truth.subjectParameters['ubar'].reshape(-1,1)
V = model_truth.subjectParameters['V']
nvi = len(agei)

print(a)
print(a_truth)
print(b)
print(b_truth)
print(ubar)
print(ubar_truth)
print(V)
print(V_truth)
print(nvi)


# In[16]:

detV = sp_linalg.det(V)
print(detV)


# In[27]:

lmbda = model_truth.lmbda

aRa = np.sum(np.square(a / lmbda))
print(aRa)
aRb = np.dot(b.T, a / np.square(lmbda))
print(aRb)


# In[30]:

print(np.dot(agei, np.matmul(yi, a / np.square(lmbda))))
print(np.sum(agei) * aRb)
print(np.dot(agei, np.matmul(yi, a / np.square(lmbda))) - np.sum(agei) * aRb)

print(np.dot(np.sum(yi, axis=0), a / np.square(lmbda)))
print(nvi * aRb)
print(np.dot(np.sum(yi, axis=0), a / np.square(lmbda)) - nvi * aRb)

invSi_mui = np.vstack(( np.dot(agei, np.matmul(yi, a / np.square(lmbda))) - np.sum(agei) * aRb,
                        np.dot(np.sum(yi, axis=0), a / np.square(lmbda)) - nvi * aRb ))

print(invSi_mui)


# In[ ]:

# code this the slow way to check
for 


# In[11]:

# project an individual onto the correct trajectory model
i = 10
idx = subject==i
agei = age[idx]
yi = y[idx,:]
model_truth.project(agei,yi)


# In[32]:

print(alpha_truth[i],beta_truth[i])


# In[11]:

model_nocorr = ps.LinearPSModel(data, y=y, correlationType='unstructured')

model_nocorr.fit()


# In[12]:

data = pd.DataFrame({'subjectIDps':[0,0,0,1,1], 
                     'age':[50,60,70,55,65]})

numVisitsPerSubject = data.groupby('subjectIDps').size()

alpha = np.array((1.1, 1.3)).reshape(-1,1)
beta = np.array((-5, 7)).reshape(-1,1)
ps = np.repeat(alpha, numVisitsPerSubject) * data['age'] +      np.repeat(beta, numVisitsPerSubject)

numBiomarkers = 5
p = np.vstack
trajectory = ps.LinearTrajectory(numBiomarkers)
trajectory.setParams()


# In[ ]:

lt = ps.LinearPS(1)


# In[ ]:

type(np.ones(3).reshape(-1,1))


# In[ ]:

type(np.array((1,0)).reshape(-1,1))


# In[ ]:

np.matrix('1;0')


# In[ ]:

np.matrix(np.ones(3).reshape(-1,1))


# In[ ]:

np.matrix(np.eye(2))


# In[ ]:

['a','b'] + [None]


# In[ ]:

tmp = np.array((1,2)).reshape(-1,1)


# In[ ]:

tmp.transpose() * tmp


# In[ ]:

hodu = np.matrix([(1,2),(3,4)])


# In[ ]:

hodu[1,0]


# In[ ]:

np.concatenate((tmp,tmp),axis=1)


# In[ ]:

np.sum(np.hstack((tmp,tmp)))


# In[ ]:

p


# In[ ]:

np.matrix([16,25]).shape


# In[ ]:

a = np.matrix([1,4])
print("a: %s" % (a.T.shape,))


# In[ ]:



