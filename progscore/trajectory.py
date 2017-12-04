from abc import ABCMeta, abstractmethod
import numpy as np

class Trajectory(metaclass=ABCMeta):
    def __init__(self, nb=1, **kwargs):
        if nb<1:
            raise ValueError('Number of biomarkers must be a positive integer')

        # numBiomarkers is a scalar
        self.numBiomarkers = nb

        # params is a numpy array, #biomarkers-by-#parameters
        self.params = np.ones((nb,self.__class__.numParams))

        # fixed_p_mask is a bool numpy array, #biomarkers-by-#parameters
        self.fixed_p_mask = np.zeros_like(self.params, dtype=bool)

    @staticmethod
    @abstractmethod
    def predictStatic(s,p):
        pass

    @abstractmethod
    def flip(self):
        pass

    @abstractmethod
    def scale(self, mu, sdev):
        pass

    @staticmethod
    @abstractmethod
    def isdegenerate(p,k=None):
        # check if parameters p would yield non-monotonic or
        # otherwise inappropriate functions
        pass

    #@abstractmethod
    #def copy(self):
    #    pass

    def predict(self,s,k=None):
        if k is None:
            p = self.params
        else:
            if k<self.numBiomarkers:
                raise ValueError('k must be smaller than number of biomarkers')
            p = self.params[k,:]

        return self.predictStatic(s,p)

    def setParams(self,p,k=None):
        if k is None:
            if not p.shape==(self.numBiomarkers, self.__class__.numParams):
                raise ValueError('Specified parameters do not have the correct dimensions')
            if np.any(self.fixed_p_mask):
                if not all(self.params[self.fixed_p_mask]==p[self.fixed_p_mask]):
                    raise ValueError('Cannot change fixed parameters')
            if all(self.__class__.isdegenerate(p)):
                raise ValueError('Cannot set parameters to degenerate values')
            self.params = p
        else:
            if k<self.numBiomarkers:
                raise ValueError('k must be smaller than number of biomarkers')
            if not len(p)==self.__class__.numParams:
                raise ValueError('Specified parameters do not have correct dimensions!')
            if np.any(self.fixed_p_mask[k,:]):
                if not all(self.params[k,self.fixed_p_mask[k,:]]==p[k,self.fixed_p_mask[k,:]]):
                    raise ValueError('Cannot change fixed parameters')
            tmp = self.params
            tmp[k,:] = p
            if np.all(self.__class__.isdegenerate(tmp)):
                raise ValueError('Cannot set parameters to degenerate values')
            self.params[k,:] = p

    def setFixedParams(self,fpm,fp,k):
        if k<self.numBiomarkers:
            raise ValueError('k must be smaller than number of biomarkers')
        pass

    def getTotalNumParams(self):
        return self.__class__.numParams * self.numBiomarkers

    def plot(self, srange, **kwargs):
        y = self.predict(srange)

    def plotk(self, srange, k , **kwargs):
        if k<self.numBiomarkers:
            raise ValueError('k must be smaller than number of biomarkers')
        y =self.predict(srange, k)
