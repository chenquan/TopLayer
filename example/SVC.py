from toplayer import TopLayer
from sklearn.svm import SVC
from sklearn.externals import joblib
import os
import tensorflow as tf
import numpy as np


class SVCLayer(TopLayer):
    def model(self, x, y_true_v, path):

        if os.path.isfile(path):
            svc = joblib.load(path)
        else:
            svc = SVC()

        svc.fit(x, y_true_v)
        joblib.dump(svc, path)
        return svc.predict(x)


@SVCLayer(name="svc", path="model/svc.m")
def SVC_loss(x, y_true, y_pred):
    pass
