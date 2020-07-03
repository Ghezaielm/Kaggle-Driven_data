import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split,cross_val_score

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint


class richeter(): 
    
    def __init__(self): 
        self.train = "/home/jaz/train_vals.csv"
        self.test = "/home/jaz/test_vals.csv"
        self.sub = "/home/jaz/Bureau/Richeter_prediction/submission_format.csv"

    def loadData(self): 
        self.train = pd.read_csv(self.train)
        self.pred = pd.read_csv(self.test)
        self.labels = self.train["label"]
        self.train = self.train.filter([i for i in self.train.columns
                                       if i not in ["building_id","label"]])
        self.pred = self.pred.filter([i for i in self.pred.columns
                                       if i not in ["building_id","label"]])
        self.sub = pd.read_csv(self.sub)
        
    def processing(self): 
        # Scale the data to 0 1
        scaler = MinMaxScaler(feature_range=(0,1))
        self.train = scaler.fit_transform(self.train)
        self.pred = scaler.fit_transform(self.pred)
        # one hot encode the labels
        self.labels = pd.get_dummies(self.labels)
        # log transformation
        self.train,self.pred = np.log(self.train+1),np.log(self.pred+1)
        # data split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
        self.train, self.labels.values, test_size=0.33, random_state=42)
        
    def setModel(self):
        # Deep NN classifier
        self.model = Sequential()
        # We linearly encode the signal
        self.model.add(Dense(150, activation="linear"))
        # Then add a sparsity constraint to prevent overfitting
        self.model.add(Dropout(0.1))
        # Low signals are not learnt
        self.model.add(Dense(100, activation="relu"))
        # Encode information 
        self.model.add(Dense(80, activation="linear"))
        # Low signals are not learnt
        self.model.add(Dense(50, activation="relu"))
        # Encode information
        self.model.add(Dense(25, activation="linear"))
        # Encode robust informations
        self.model.add(Dense(12, activation="linear"))
        # Signal to probabilities
        self.model.add(Dense(self.y_train.shape[1],activation='softmax'))
        
        # Compile the model
        self.model.compile(optimizer='adam', 
                           loss='binary_crossentropy',metrics=["accuracy"])
        
        # Set a callback to monitor model relevance
        self.filepath="Richeter_damage_DNN.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, 
                                         save_best_only=True, mode='min')
        self.callbacks_list = [checkpoint]
        
    def launchModel(self):
        # Training
        self.history= self.model.fit(self.X_train, self.y_train, epochs=100,
                                 batch_size=100,verbose=1,
                                     validation_data=(self.X_test,self.y_test), 
                                    callbacks=self.callbacks_list)
        
        ## Results visualization
        # Create canvas
        f = plt.figure(figsize=(10,10))
        f.suptitle("Metrics")
        ax = f.add_subplot(221)
        ax1 = f.add_subplot(222)
        ax2 = f.add_subplot(223)
        ax3 = f.add_subplot(224)
        # Plot the results
        ax.plot(self.history.history["loss"])
        ax.set_title("train loss")
        ax1.plot(self.history.history["accuracy"])
        ax1.set_title("train accuracy")
        ax2.plot(self.history.history["val_loss"])
        ax2.set_title("test loss")
        ax3.plot(self.history.history["val_accuracy"])
        ax3.set_title("test accuracy")
        
        # Predict classes for the query set
        self.model.load_weights(Richeter_damage_DNN)
        self.preds = self.model.predict(self.pred)
        
        # Save the predictions
        self.sub["damage_grade"]=np.argmax(self.preds,axis=1)+1
        self.sub.to_csv("out.csv",index=False)
        
        
        
        
        
    
        
_ = richeter()
_.loadData()
_.processing()
_.setModel()
_.launchModel()
