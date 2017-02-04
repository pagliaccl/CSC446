from sklearn.datasets import load_svmlight_file
import numpy as np
from Pla import Pla
from LinearSVM import SVM

x_train, y_train = load_svmlight_file("/Users/Pagliacci/Desktop/MachineLearing/a7a.train.csv", n_features=123)
# x_test, y_test = load_svmlight_file("/Users/Pagliacci/Desktop/MachineLearing/a7a.test.csv", n_features=123)
# print y_test
# print x_train.shape
# a=()

# print (np.zeros(123)+np.transpose(x_train[2])).shape


# pla=Pla(2)
# pla.train("/Users/Pagliacci/Desktop/MachineLearing/a7a.train.csv")
# print pla.test("/Users/Pagliacci/Desktop/MachineLearing/a7a.test.csv")

svm=SVM(100,c=4)
svm.train("/Users/Pagliacci/Desktop/MachineLearing/a7a.train.csv",)
print svm.test("/Users/Pagliacci/Desktop/MachineLearing/a7a.test.csv")
