from __future__ import print_function, absolute_import

import numpy as np

from sklearn.model_selection import train_test_split



class LogisticRegression:
    def __init__(self,data,target,test_size=0.2,val_size=0.1,regularization=None,f='sigmoid'):
        self.data       = data
        
        self.ts_size    = test_size
        self.val_size   = val_size

        self.target     = target
        self.reg        = regularization
        self.f          = f

        self.X          = self.data[:,:self.target]
        self.Y          = self.data[:,self.target]

    
    def func(self,z):
        if(self.f == 'sigmoid'):
            return 1/(1 + np.exp(-z))
        else:
            return np.tanh(z)
    

    def loss(self,Y_true,Y_pred):
        loss            = np.mean(Y_true * np.log(Y_pred + 1e-1) + (1-Y_true) * np.log(1-Y_pred + 1e-1))

        return -loss
    

    def train(self,lr=0.1,n_epochs=1000,weights=None,batch_size=21):
        self.tr_loss    = []
        self.val_loss   = []
        self.ts_loss    = []

        self.tr_acc     = []
        self.val_acc    = []
        self.ts_acc     = []

        self.tr_pr      = []
        self.val_pr     = []
        self.ts_pr      = []

        self.tr_re      = []
        self.val_re     = []
        self.ts_re      = []

        self.tr_f1      = []
        self.val_f1     = []
        self.ts_f1      = []

        self.epochs     = n_epochs
        self.lr         = lr
        self.batch      = batch_size

        bias_term       = np.ones((self.X.shape[0],1))
        self.data       = np.hstack((self.X,bias_term))

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, 
                                                                                self.Y, 
                                                                                test_size=self.ts_size,
                                                                                random_state=42,
                                                                                stratify=self.Y)

        if(self.val_size>0):
            self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(self.X_train, 
                                                                                    self.Y_train, 
                                                                                    test_size=self.val_size,
                                                                                    random_state=42,
                                                                                    stratify=self.Y_train)
        
        self.n_s,self.n_f       = self.X_train.shape

        if(weights is None):
            self.weights = np.ones(self.n_f)
            self.weights += 1
        else:
            self.weights = weights


        for _ in range(n_epochs):

            if(self.val_size>0):
                ## Validation set performance metrics
                self.Y_val_pred = self.func(np.dot(self.X_val,self.weights))
                self.Y_val_pred = np.where(self.Y_val_pred >= 0.5, 1, 0)

                self.val_loss.append(self.loss(self.Y_val,self.Y_val_pred))

                a,p,r,f         = self.calculate_metrics(self.Y_val,self.Y_val_pred)
                self.val_acc.append(a)
                self.val_pr.append(p)
                self.val_re.append(r)
                self.val_f1.append(f)


            ## Training set performance metrics
            z = np.dot(self.X_train,self.weights)
            self.Y_pred = self.func(z)
            self.Y_pred = np.where(self.Y_pred >= 0.5, 1, 0)

            self.tr_loss.append(self.loss(self.Y_train,self.Y_pred))

            a,p,r,f     = self.calculate_metrics(self.Y_train, self.Y_pred)
            self.tr_acc.append(a)
            self.tr_pr.append(p)
            self.tr_re.append(r)
            self.tr_f1.append(f)


            inds        = np.arange(self.n_s)
            np.random.shuffle(inds)

            for i in range(0,self.n_s,self.batch):
                j               = min(i+self.batch,self.n_s)
                
                X_batch    = self.X_train[i:j]
                Y_batch    = self.Y_train[i:j]

                z_batch         = np.dot(X_batch,self.weights)
                y_pred          = self.func(z_batch)

                gradient        = np.dot(X_batch.T,(y_pred-Y_batch))/self.batch

                self.weights   -= self.lr * gradient

    
    def predict(self):
        self.Y_test_pred    = self.func(np.dot(self.X_test,self.weights))
        self.ts_loss        = self.loss(self.Y_test_pred,self.Y_test)

        a,p,r,f         = self.calculate_metrics(self.Y_test,self.Y_test_pred)
        self.ts_acc.append(a)
        self.ts_pr.append(p)
        self.ts_re.append(r)
        self.ts_f1.append(f)


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