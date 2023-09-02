from __future__ import print_function, absolute_import

import numpy as np

from sklearn.model_selection import train_test_split



class LogisticRegression:
    def __init__(self,data,target,test_size=0.2,val_size=0,regularization=None):
        self.data       = data
        self.n_s        = self.data.shape[0]
        self.n_f        = self.data.shape[1]
        
        self.ts_size    = test_size
        self.val_size   = val_size

        self.target     = target
        self.epochs     = []
        self.reg        = regularization

        self.X          = self.data[:,:self.target]
        self.Y          = self.data[:,self.target]

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=self.ts_size, shuffle=True,
                                                                                random_state=42, stratify=self.Y)
        
        self.X_train, self.X_val, self.Y_train, self.Y_val   = train_test_split(self.X_train, self.Y_train, test_size=self.val_size,
                                                                                random_state=42)

        self.tr_loss    = []
        self.val_loss   = []
        self.ts_loss    = []

        self.tr_acc     = []
        self.val_acc    = []
        self.ts_acc     = []

    
    def sigm(self,z):
        return 1/(1 + np.exp(-z))
    

    def loss(self,Y_true,Y_pred):
        eps = 1e-10

        Y_pred          = np.clip(Y_pred,eps,1-eps)
        loss            = np.mean(Y_true * np.log(Y_pred) + (1-Y_true) * np.log(1-Y_pred))

        return -loss
    

    def train(self,lr=0.1,n_epochs=1000,weights=None):
        self.lr         = lr

        if(weights is None):
            self.weights = np.zeros(self.X_train.shape[1])
        else:
            self.weights = weights


        for n in range(n_epochs):
            self.epochs.append(n+1)

            for i in range(self.X_train.shape[0]):
                xi      = self.X_train[i]
                yi      = self.Y_train[i]

                z       = np.dot(xi,self.weights)
                y_pred  = self.sigm(z)

                self.weights   -= self.lr * (np.dot(xi.T,(y_pred - yi)))

            self.Y_pred     = self.sigm(np.dot(self.X_train,self.weights))

            self.tr_loss.append(self.loss(self.Y_train,self.Y_pred))
            self.tr_acc.append(self.calculate_metrics(self.Y_train, self.Y_pred)[0])

            self.Y_val_pred = self.sigm(np.dot(self.X_val,self.weights))

            self.val_loss.append(self.loss(self.Y_val,self.Y_val_pred))
            self.val_acc.append(self.calculate_metrics(self.Y_val, self.Y_val_pred)[0])

    
    def predict(self):
        self.Y_test_pred    = self.sigm(np.dot(self.X_test,self.weights))
        self.ts_loss        = self.loss(self.Y_test_pred,self.Y_test)


    def calculate_metrics(self, Y_true, Y_pred):
        Y_pred_binary = np.where(Y_pred >= 0.5, 1, 0)

        tp = np.sum((Y_true == 1) & (Y_pred_binary == 1))
        tn = np.sum((Y_true == 0) & (Y_pred_binary == 0))
        fp = np.sum((Y_true == 0) & (Y_pred_binary == 1))
        fn = np.sum((Y_true == 1) & (Y_pred_binary == 0))
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return accuracy, precision, recall, f1_score