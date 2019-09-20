from sklearn.linear_model import Lars, Lasso
from sklearn.externals import joblib
import os

from toplayer.toplayer import TopLayer


class LarsLayer(TopLayer):

    def model(self, x, y_true_v, path):
        print(type(x))
        print(x.shape, y_true_v.shape)
        if os.path.isfile(path):
            lars = joblib.load(path)
        else:
            lars = Lars()
        lars.fit(x, y_true_v)
        joblib.dump(lars, path)
        return lars.predict(x)


class LassoLayer(TopLayer):
    def model(self, x, y_true_v, path):
        lasso = Lasso()
        lasso.fit(x, y_true_v)
        return lasso.predict(x)
