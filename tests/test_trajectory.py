from progscore import LinearTrajectory
import unittest
import numpy as np
from ddt import ddt, data, unpack

@ddt
class TestLinearTrajectoryOneBiomarker(unittest.TestCase):
    def setUp(self):
        self.lt = LinearTrajectory()

        p = np.array((1,2))
        p = p[np.newaxis,:]
        self.lt.setParams(p)

    def test_getTotalNumParams(self):
        self.assertEqual(self.lt.getTotalNumParams(),2)

    def test_predictStatic(self):
        s = np.array((-10,0,10)).reshape(-1,1)

        p = np.array((1,2))
        p = p[np.newaxis,:]

        y, J, JJ = LinearTrajectory.predictStatic(s,p)
        self.assertTrue((y==s+2).all())

    def test_predict(self):
        s = np.array((-10,0,10)).reshape(-1,1)
        y, J, JJ = self.lt.predict(s)
        self.assertTrue((y==s+2).all())

    def test_setParams_degenerate(self):
        p = np.array((0,2))
        p = p[np.newaxis,:]
        self.assertRaises(ValueError, self.lt.setParams, p)

    def test_scale_identity(self):
        p = self.lt.params
        self.lt.scale(0,1)
        self.assertTrue((self.lt.params==p).all())

    def test_flip_identity(self):
        p = self.lt.params
        self.lt.flip().flip()
        self.assertTrue((self.lt.params==p).all())
