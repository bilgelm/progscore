from .trajectory import Trajectory
import numpy as np

class LinearTrajectory(Trajectory):
    numParams = 2

    @staticmethod
    def isdegenerate(p,k=None):
        # linear trajectory is non-monotonic if slope=0
        if k is None:
            return p[:,0]==0
        else:
            return p[k,0]==0

    @staticmethod
    def predictStatic(s,p):
        numParams = LinearTrajectory.numParams

        if not p.shape[1]==numParams:
            raise ValueError('Number of parameters must be 2 for the linear trajectory!')

        a = p[:,0]
        b = p[:,1]

        y = np.outer(s,a) + b.T

        J = np.zeros((len(s), numParams))
        dyda = s.flatten()
        dydb = 1
        J[:,0] = dyda
        J[:,1] = dydb

        JJ = np.zeros((len(s), numParams, numParams))

        return (y, J, JJ)

    def flip(self):
        self.params[:,0] = -self.params[:,0]

    def scale(self,mu,sdev):
        if sdev<=0:
            raise ValueError('Standard deviation must be positive')
        self.params[:,1] = self.params[:,1] + mu*self.params[:,0]
        self.params[:,0] = sdev*self.params[:,0]
