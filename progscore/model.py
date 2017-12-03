from abc import ABCMeta, abstractmethod
import numpy as np

class PSModel(metaclass=ABCMeta):
    def __init__(self,
                 correlationType='identity',
                 maxIterations=100,
                 doMultiLambda=True):

        self.maxIterations = maxIterations
        self.doMultiLambda = doMultiLambda
        self.correlationType = correlationType

    @abstractmethod
    def fit(self):
        return self

class NonlinearPSModel(PSModel):
    def __init__(self, trajectory):
        super()
        self.trajectory = trajectory

    def fit(self):
        return self

class LinearPSModel(PSModel):
    def fit(self):
        subjectIDps = self.subjectIDps
        age = self.age
        return self

    def expectation(self, ):
        # E-step
        trace_term1 = 0
        detV = np.linalg.det(V)

        if self.correlationType == 'identity':
            aRa = np.sum((a / lmbda)**2)
        else:
            count = 0
            inv_tS_a = np.linalg.lstsq(tS, a)
            aRa = np.sum(inv_tS_a**2)

        for i in range(numSubjects):
            idx = subjectIDps==i
            agei = age[idx]
            yi = y[idx,:].T
            nvi = numVisitsPerSubject[i]

            if self.correlationType == 'identity':
                invSi_mui = np.vstack((agei, np.ones_like(agei))) * \
                            np.sum(yi * a / (lmbda**2), axis=1) \
                            - np.sum(a * b / (lmbda**2)) * np.array((np.sum(agei), nvi))
            else:
                invSi_mui = np.zeros((2,1))
                for j in range(nvi):
                    yij = y[count,:].flatten()
                    q = np.array((age[count],1)).reshape(-1,1)
                    invSi_mui += q.dot(inv_tS_a.T).dot(np.linalg.lstsq(tS,yij-b))
                    count += 1

            Sigmai = aRa * np.matrix([nvi, -np.sum(agei)],
                                     [-np.sum(agei), np.sum(agei**2)]) + V/detV
            Sigmai /= np.linalg.det(Sigmai)

            Sigma_00[i] = Sigmai[0,0]
            Sigma_01[i] = Sigmai[0,1]
            Sigma_11[i] = Sigmai[1,1]

            trace_term1 += Sigma_00[i] * np.sum(agei**2) + \
                           2*Sigma_01[i] * np.sum(agei) + \
                           Sigma_11[i] * nvi
            alpha[i], beta[i] = Sigmai * (invSi_mui + np.linalg.lstsq(V, ubar))

        return self

    def maximization(self, ):
        sum_s = np.sum(self.ps)
        sum_ssq = np.sum(self.ps**2)
        sum_ys = np.sum(self.y * self.ps, axis=0).flatten() # K-by-1

        a = (sum_ys - sum_s * ybar) / \
            (sum_ssq + trace_term1 - sum_s**2 / numSubjVisits)
        b = ( ( sum_ssq + trace_term1 ) * ysum - sum_s * sum_ys ) / \
            (numSubjVisits * ( sum_ssq + trace_term1 ) - sum_s**2)

        ubar[0] = np.mean(alpha)
        ubar[1] = np.mean(beta)

        V = alpha-ubar[0]

        return self
