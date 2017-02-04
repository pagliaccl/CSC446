import numpy as np
from sklearn.datasets import load_svmlight_file

class SVM:
    def __init__(self, maxit=100,c=1):
        self.maxIt = maxit
        self.shape=tuple()
        self.curit = 1
        self.error=0
        self.w=None
        self.C=c



    def load(self, dir, feat=123):
        x_train, y_train = load_svmlight_file(dir, n_features=feat)
        return (x_train, y_train)

    def activation(self, y):

        return 1 if y>=0 else -1

    def train(self, dir):
        d, y=self.load(dir)
        self.shape= d.shape
        x = d.todense()
        x=np.append(x, np.ones(self.shape[0]).reshape((self.shape[0],1)),1)
        self.shape = x.shape
        self.w = np.reshape(np.zeros(self.shape[1]), (1, self.shape[1]))

        while True:
            for j in range(self.shape[0]):
                yhat=self.activation(int(np.dot(self.w, np.transpose(x[j]))))
                self.w= self.w + ((self.shape[0]*self.C/self.curit)*(np.sign(y[j]-yhat) * x[j])) - (2./self.curit)*self.w
                self.curit+=1
                if self.curit>self.maxIt: break
            if self.curit > self.maxIt: break


    def test(self, dir):
        result=0
        x, y = self.load(dir)
        x=x.todense()
        x=np.append(x, np.ones(x.shape[0]).reshape((x.shape[0],1)),1)

        for i in range(len(y)):
            yhat = self.activation(int(np.matmul(self.w, np.transpose(x[i]))))
            if y[i]== yhat:
                result+=1

        return result/float(len(y))