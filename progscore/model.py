from abc import ABCMeta, abstractmethod
import numpy as np
import scipy as sp
from scipy import linalg as sp_linalg
import pandas as pd # ?

from .lineartrajectory import LinearTrajectory

class PSModel(metaclass=ABCMeta):
    def __init__(self,
                 data,
                 biomarkerLabels=None,
                 y=None,
                 correlationType='identity',
                 maxIterations=100,
                 tol=1e-6):

        if not ((biomarkerLabels is None) ^ (y is None)):
            raise ValueError('Either biomarkerLabels or y must be specified, not both')

        required_cols = ['subjectIDps','age'] #,'currdx']
        if not biomarkerLabels is None:
            if len(biomarkerLabels) < 2:
                raise ValueError('There must be at least two biomarkers')
            required_cols += biomarkerLabels
        else:
            if y.shape[1] < 2:
                raise ValueError('There must be at least two biomarkers')

        # data must have the following columns: subjectIDps, age, currdx
        for col in required_cols:
            if not col in data.columns:
                raise ValueError('Required column '+col+' is not present in data')

        if data.duplicated(subset=['subjectIDps','age']).any():
            raise ValueError('There are duplicate visit entries in data')

        self.numSubjects = data['subjectIDps'].nunique()
        if not (data['subjectIDps'].min()==0) and \
               (data['subjectIDps'].max()==self.numSubjects-1):
            raise ValueError('Subject IDs must be consecutive integers, numbered starting with 0')

        if not correlationType in ['identity','unstructured']:
            raise NotImplementedError('Correlation of type '+correlationType+'is not implemented')

        if maxIterations<1:
            raise ValueError('Maximum iteration number must be at least 1')

        # age must be increasing within each subject ID
        #data.sort_values(['subjectIDps','age'], ascending=[True, True], inplace=True)
        #data.reset_index(drop=True, inplace=True)

        if not biomarkerLabels is None:
            y = data.as_matrix(biomarkerLabels)
            data = data.drop(biomarkerLabels, axis=1)

        self.data = data
        self.y = y

        self.numVisitsPerSubject = data.groupby('subjectIDps').size()
        self.numSubjVisits, self.numBiomarkers = self.y.shape

        self.maxIterations = maxIterations
        #self.doMultiLambda = doMultiLambda
        self.correlationType = correlationType

        self.subjectVariables = {'alpha': np.ones(self.numSubjects).reshape(-1,1),
                                 'beta': np.zeros(self.numSubjects).reshape(-1,1)}
        self.subjectParameters = {'ubar': np.array((1,0)).reshape(-1,1),
                                  'V': np.eye(2)} # should it be a matrix?

        self.lmbda = np.ones(self.numBiomarkers).reshape(-1,1)
        self.C = np.eye(self.numBiomarkers)
        self.R = np.eye(self.numBiomarkers)

        # initialize with linear trajectory for each marker
        self.trajectory = LinearTrajectory(self.numBiomarkers)

        self.tol = tol
        self.converged = False
        self.loglik = -np.inf

    @property
    def ps(self):
        ps = np.repeat(self.subjectVariables['alpha'],
                       self.numVisitsPerSubject) * self.data['age'].as_matrix() + \
             np.repeat(self.subjectVariables['beta'],
                       self.numVisitsPerSubject)
        return ps

    @property
    def y_pred(self):
        return self.predict(self.ps)

    def predict(self, s, k=None):
        return self.trajectory.predict(s, k=k)

    @abstractmethod
    def fit(self):
        return self

    @abstractmethod
    def project(self, subject):
        return self

    def standardize(self):
        # ensure that on average, ps increases with age
        if self.subjectParameters['ubar'][0] < 0:
            self.trajectory.flip()
            self.subjectParameters['ubar'] *= -1
            self.subjectVariables['alpha'] *= -1
            self.subjectVariables['beta'] *= -1

        # Normalize ps
        ps = self.ps
        mu = np.mean(ps)
        sdev = np.std(ps)
        self.subjectVariables['alpha'] /= sdev
        self.subjectVariables['beta'] = (self.subjectVariables['beta'] - mu) / sdev
        self.subjectParameters['ubar'] = (self.subjectParameters['ubar'] - np.array((0,mu)).reshape(-1,1)) / sdev
        self.subjectParameters['V'] /= sdev**2
        self.trajectory.scale(mu, sdev)

    def getNumNoiseParams(self):
        if self.correlationType=='identity':
            numNoiseParams = self.numBiomarkers
        elif self.correlationType=='unstructured':
            numNoiseParams = self.numBiomarkers*(self.numBiomarkers+1)/2
        else:
            raise NotImplementedError()

        return numNoiseParams

    def getNumSubjectParams(self):
        numSubjectParams = 2 + 3
        return numSubjectParams

    def AIC(self):
        return 2*(self.trajectory.getTotalNumParams() + \
                  self.getNumNoiseParams() + \
                  self.getNumSubjectParams() - self.loglik)

class NonlinearPSModel(PSModel):
    def __init__(self,
                 data,
                 trajectory,
                 biomarkerLabels=None,
                 y=None, **kwargs):
        super(data,
              biomarkerLabels=biomarkerLabels,
              y=y, **kwargs).__init__()
        self.trajectory = trajectory

    def fit(self):
        return self

class LinearPSModel(PSModel):
    def fit(self):
        #subjectIDps = self.data['subjectIDps']
        #age = self.data['age']
        #a = self.trajectory.params[:,0]
        #b = self.trajectory.params[:,1]

        loglik_old = self.loglik

        for itr in range(self.maxIterations):
            trace_term1, Sigma_00, Sigma_01, Sigma_11 = \
                                              self._updateSubjectVariables()
            self._updateTrajectoryParameters(trace_term1,
                                             Sigma_00, Sigma_01, Sigma_11)

            self.loglik = self._compute_loglik()
            #if self.loglik < loglik_old:
            #    break

            if np.absolute(self.loglik/loglik_old - 1) < self.tol:
                self.converged = True
                break

            loglik_old = self.loglik

        self.standardize()

        return self

    def project(self, agei, yi):
        if not len(agei)==yi.shape[0]:
            raise ValueError('Length of agei vector and number of rows in yi must be the same')
        if not yi.shape[1]==self.numBiomarkers:
            raise ValueError('Number of columns of yi and number of biomarkers used to train model must be the same')

        a = self.trajectory.params[:,0].reshape(-1,1)
        b = self.trajectory.params[:,1].reshape(-1,1)
        ubar = self.subjectParameters['ubar'].reshape(-1,1)
        V = self.subjectParameters['V']
        nvi = len(agei)

        #print(a.shape)
        #print(b.shape)
        #print(ubar.shape)
        #print(V.shape)
        #print(agei)
        #print(yi.shape)

        detV = sp_linalg.det(V)

        if self.correlationType == 'identity':
            lmbda = self.lmbda

            aRa = np.sum(np.square(a / lmbda))
            aRb = np.dot(b.T, a / np.square(lmbda))

            invSi_mui = np.vstack(( np.dot(agei, np.matmul(yi, a / np.square(lmbda))) - np.sum(agei) * aRb,
                                    np.dot(np.sum(yi, axis=0), a / np.square(lmbda)) - nvi * aRb ))
        else:
            R = self.R
            aRa = np.asscalar(np.dot(a.T,sp_linalg.solve(R,a,assume_a='pos')))

            invSi_mui = np.zeros((2,1))
            for j in range(nvi):
                yij = yi[j,:].reshape(-1,1)
                q = np.array((agei[j],1)).reshape(-1,1)
                invSi_mui += np.matmul(np.dot(q,a.T), sp_linalg.solve(R,yij-b,assume_a='pos'))

        Sigmai = aRa * np.matrix([ (nvi, -np.sum(agei)),
                                   (-np.sum(agei), np.sum(np.square(agei))) ]) + V/detV
        Sigmai /= sp_linalg.det(Sigmai)

        #print(a.shape)
        #print(b.shape)
        #print(lmbda.shape)

        alphai, betai = np.matmul(Sigmai, invSi_mui + sp_linalg.solve(V,ubar,assume_a='pos'))
        #print(alphai.shape)
        #print(betai.shape)
        #print(Sigmai.shape)
        return (alphai, betai, Sigmai)

    def _updateSubjectVariables(self):
        # E-step
        subjectIDps = self.data['subjectIDps'].as_matrix()
        age = self.data['age'].as_matrix()

        trace_term1 = 0
        Sigma_00 = Sigma_01 = Sigma_11 = np.zeros(self.numSubjects)

        for i in range(self.numSubjects):
            idx = subjectIDps==i
            agei = age[idx]
            yi = self.y[idx,:]
            nvi = len(agei)

            alphai, betai, Sigmai = self.project(agei, yi)

            self.subjectVariables['alpha'][i] = alphai
            self.subjectVariables['beta'][i] = betai

            Sigma_00[i] = Sigmai[0,0]
            Sigma_01[i] = Sigmai[0,1]
            Sigma_11[i] = Sigmai[1,1]

            trace_term1 += Sigma_00[i] * np.sum(np.square(agei)) + \
                           2*Sigma_01[i] * np.sum(agei) + \
                           Sigma_11[i] * nvi

        return (trace_term1, Sigma_00, Sigma_01, Sigma_11)

    def _updateTrajectoryParameters(self, trace_term1,
                                    Sigma_00, Sigma_01, Sigma_11):
        alpha = self.subjectVariables['alpha']
        beta = self.subjectVariables['beta']
        ps = self.ps

        sum_s = np.sum(ps)
        sum_ssq = np.sum(np.square(ps))
        sum_ys = np.matmul(self.y.T, ps).reshape(-1,1) # K-by-1

        ybar = np.mean(self.y,axis=0).reshape(-1,1)
        ysum = np.sum(self.y,axis=0).reshape(-1,1)

        a = (sum_ys - sum_s * ybar) / \
            (sum_ssq + trace_term1 - sum_s**2 / self.numSubjVisits)
        b = ( ( sum_ssq + trace_term1 ) * ysum - sum_s * sum_ys ) / \
            (self.numSubjVisits * ( sum_ssq + trace_term1 ) - sum_s**2)

        # handle fixed trajectory parameters - TODO

        # update trajectory parameters
        self.trajectory.setParams(np.hstack((a,b)))

        self.subjectParameters['ubar'][0] = np.mean(alpha)
        self.subjectParameters['ubar'][1] = np.mean(beta)
        '''
        # there's something wrong here
        tmp = np.hstack((alpha-self.subjectParameters['ubar'][0],
                         beta-self.subjectParameters['ubar'][1]))
        self.subjectParameters['V'] = np.dot(tmp.T, tmp) + \
            np.matrix([(np.sum(Sigma_00), np.sum(Sigma_01)),
                       (np.sum(Sigma_01), np.sum(Sigma_11))]) / self.numSubjects
        '''
        if self.correlationType=='identity':
            self.lmbda = (np.sum(np.square(self.y-self.y_pred), axis=0).reshape(-1,1) + \
                          trace_term1 * np.square(a)) / self.numSubjVisits
            self.R = np.diag(np.square(self.lmbda).flatten())
        elif self.correlationType=='unstructured':
            self.R = (np.matmul((self.y-self.y_pred).T, self.y-self.y_pred) + \
                      np.matmul(a,a.T)) / self.numSubjVisits # np.outer check
            self.lmbda = np.sqrt(np.diag(self.R))
            self.C = self.R / np.matmul(self.lmbda,self.lmbda.T) # np.outer check
        else:
            raise NotImplementedError()

        return self

    def _compute_loglik(self):
        subjectIDps = self.data['subjectIDps'].as_matrix()
        age = self.data['age'].as_matrix()
        b = self.trajectory.params[:,1].reshape(-1,1)
        ubar = self.subjectParameters['ubar'].reshape(-1,1)
        V = self.subjectParameters['V']

        sum_yij_b = 0
        sumlogdetSigmai = 0
        sum_uiSigmaui = 0

        invV_ubar = sp_linalg.solve(V,ubar,assume_a='pos')

        for i in range(self.numSubjects):
            idx = subjectIDps==i
            agei = age[idx]
            yi = self.y[idx,:]
            nvi = len(agei)

            if self.correlationType=='identity':
                #print(yi.shape)
                #print(b.shape)
                #print(self.lmbda.shape)
                #print(np.sum(np.square((yi.T-b)/self.lmbda)))
                sum_yij_b += np.sum(np.square((yi.T-b)/self.lmbda))
            else:
                for j in range(nvi):
                    yij = yi[j,:].reshape(-1,1)
                    tmp = sp_linalg.solve(self.R,yij-b,assume_a='pos')
                    sum_yij_b += np.dot((yij-b).T, tmp)

            alphai, betai, Sigmai = self.project(agei, yi)
            _, logdetSigmai = np.linalg.slogdet(Sigmai)
            sumlogdetSigmai += logdetSigmai

            #tmp = invSi_mui + invV_ubar
            #sum_uiSigmaui += tmp.T * Sigmai * tmp
            ui = np.vstack((alphai, betai))
            #print(alphai.shape)
            #print(betai.shape)
            #print(Sigmai.shape)
            #print(sp_linalg.solve(Sigmai,ui).shape)
            sum_uiSigmaui += np.dot(ui.T, sp_linalg.solve(Sigmai,ui))

        _, logdetV = np.linalg.slogdet(V)
        _, logdetR = np.linalg.slogdet(self.R)
        loglik =  0.5*( sumlogdetSigmai - self.numSubjects*logdetV \
                       -self.numSubjVisits*self.numBiomarkers*np.log(2*np.pi) \
                       -self.numSubjVisits*logdetR ) \
                 -0.5*sum_yij_b \
                 -0.5*self.numSubjects*np.dot(ubar.T, invV_ubar) \
                 +0.5*sum_uiSigmaui
        print(loglik)
        #print(sumlogdetSigmai)
        #print(logdetV)
        #print(logdetR)
        #print(sum_yij_b)
        #print(sum_uiSigmaui)
        #print()
        return loglik
