import numpy as np
from sklearn.datasets import load_svmlight_file

class Pla:
    def __init__(self,maxit):
        self.maxIt = maxit
        self.shape=tuple()
        self.error=0
        self.w=None

    # def __init__(self, maxit=1):
    #     self.maxIt = maxit
    #     self.shape=tuple()
    #     self.error=0
    #     self.w=None




    def load(self, dir):
        x_train, y_train = load_svmlight_file("/Users/Pagliacci/Desktop/MachineLearing/a7a.train.csv", n_features=123)
        return (x_train, y_train)

    def activation(self, y):

        return 1 if y>=0 else -1


    def train(self, dir):
        d, y=self.load(dir)
        self.shape= d.shape
        self.w=np.reshape(np.zeros(self.shape[1]),(1,self.shape[1]))

        x = d.todense()
        for i in range(self.maxIt):
            for j in range(self.shape[0]):
                yhat=self.activation(int(np.dot(self.w, np.transpose(x[j]))))


                self.w= self.w + (np.sign(y[j]-yhat) * x[j])


    def test(self, dir):
        result=0
        x, y = self.load(dir)
        x=x.todense()

        for i in range(len(y)):
            yhat = self.activation(int(np.matmul(self.w, np.transpose(x[i]))))
            if y[i]== yhat:
                result+=1

        return result/float(len(y))