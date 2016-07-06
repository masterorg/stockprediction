import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
from random import random
from keras.models import Sequential  
from keras.layers.core import Dense, Activation , Dropout
from keras.layers.recurrent import LSTM



      
def hell_model(hidden_neurons, input_num):
    model =  Sequential()
    model.add(LSTM(hidden_neurons, input_dim = input_num, return_sequences=True, init='uniform'))
    model.add(Dropout(0.5))
    model.add(LSTM(hidden_neurons/5,input_dim = hidden_neurons, return_sequences=False,init='uniform'))
    model.add(Dropout(0.2))
    model.add(Dense(1, input_dim = hidden_neurons ))
    
    model.add(Activation("linear"))  
    model.compile(loss="mean_squared_error", optimizer="sgd") #rmsprop
    return model
#new model
def new_model(hidden_neurons,input_num ):
    model =  Sequential()
    model.add(LSTM(hidden_neurons, input_dim = input_num, return_sequences=False, init='uniform'))
    model.add(Dense(1, input_dim = hidden_neurons ))
    model.add(Activation("linear"))  
    model.compile(loss="mean_squared_error", optimizer="rmsprop") #rmsprop
    return model

def load(path):
    fields = ['Close']
    dataframe = pd.read_csv(path,  usecols=fields)
    #dataframe['Close'] = dataframe['Close'].astype(float)
    
    
      
    dataframe.iloc[::-1]
    
    return dataframe
    
def split_data(data, n_prev = 3):
    
    x = [] 
    y = []
    for i in range(len(data)- n_prev ):
        x.append( data.iloc[i:i+n_prev].as_matrix())
        y.append(data.iloc[i+ n_prev].as_matrix())
    
    
        
    xx = np.array(x)
    yy = np.array(y)
    
    
    
    
    
        
    
    
    
    
      
    
    return xx, yy        
        
    
def load_data(data, train_size = 0.1):
    index = int(round(len(data) * (1- train_size)))
       
    
    X_train , Y_train  = split_data(data.iloc[0:index],20)
    X_test  , Y_test   = split_data(data.iloc[index:], 20)
    
    #return data
    return X_train, Y_train , X_test, Y_test
    
def plot_data(Y_test, predicted,title='No title'):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(Y_test, label='Original data')
    plt.plot(predicted, label="Prediction data")
    plt.title(title , fontsize=21)
    #axis labels
    plt.xlabel("Days" , fontsize=18)
    plt.ylabel("Price in $" , fontsize=18)    
    #show legend
    ax.legend(loc='upper left', shadow=True)
    plt.show()

def plot_rmse(data):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title("Root-mean-square deviation" , fontsize=21)
    plt.plot(data, label="Data")
    ax.legend(loc='upper left', shadow=True)
    plt.show()
    
if __name__ == "__main__": 
    
    
    
    
    data = load("data.csv")
    
    X_train , Y_train , X_test, Y_test =  load_data(data,0.3)
    
    
    
    #make model
    model = new_model(150,X_train.shape[2])
    model.fit(X_train, Y_train,batch_size=22, nb_epoch=15, validation_split=0.02)
    
    model.evaluate(X_test,Y_test, batch_size = 300)
    predicted = model.predict(X_test) 
    predicted = np.reshape(predicted, (predicted.size,))
    
    rmse = np.sqrt(((predicted - Y_test) ** 2).mean(axis=0))
    print(rmse)
   
    
    plot_data(Y_test, predicted,"Stock prediction")
    plot_rmse(rmse)
    
   
    
   