import numpy as np
import pandas as pd



class LogisticRegression:
    def __init__(self,data,target,test_size=0.2,val_size=0):
        self.data       = data
        self.n_s        = self.data.shape[0]
        self.n_f        = self.data.shape[1]
        
        self.ts_size    = test_size
        self.val_size   = val_size

        self.target     = target

        self.X          = self.data.drop(labels=[self.target],axis=1)
        self.Y          = self.data[self.target]

        self.X_train    = self.X[:int(self.n_s * self.ts_size),:]
        self.X_test     = self.X[int(self.n_s * self.ts_size):,:]

        self.Y_train    = self.Y[:int(self.n_s * self.ts_size),:]
        self.Y_test     = self.Y[int(self.n_s * self.ts_size):,:]
        self.Y_pred     = []

        self.weights    = np.zeros(self.n_f)

        self.tr_loss    = []
        self.val_loss   = []
        self.ts_loss    = []

    
    def sigm(self,z):
        return 1/(1 + np.exp(-z))
    

    def loss(self,Y_true,Y_pred):
        eps = 1e-10

        Y_pred          = np.clip(Y_pred,eps,1-eps)
        loss            = np.mean(Y_true * np.log(Y_pred) + (1-Y_true) * np.log(1-Y_pred))

        return -loss
    

    def train(self,lr=0.2,n_epochs=1000,bias=0):
        self.lr         = lr
        self.bias       = bias

        for n in range(n_epochs):
            self.Y_pred = []

            for i in range(self.n_s):
                xi      = self.X_train[i]
                yi      = self.Y_train[i]

                z       = np.dot(xi,self.weights) + self.bias
                y_pred  = self.sigm(z)

                self.Y_pred.append(y_pred)

                self.weights -= self.lr * ((y_pred - yi) * xi)
                self.bias    -= self.lr * (y_pred - yi)

            self.tr_loss.append(self.loss(self.Y_train,self.Y_pred))