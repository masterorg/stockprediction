# stockprediction
making neural Network for stock price prediction


Hole project is written in Python programming language. Beside basic libraries, I used: 
  pandas - for csv file manipulation 
  Theano - library for  basic mathematical operations inside neural network (its only for usage of Keras)
  Keras  - wrapper for Theano library (it does all calculations for me)
  
  
  The idea behinde this project is to use LSTM (Long - Short - Term - Memory) to try and find connection between time series of our data.
  As data source i used Apple historical data prices downlaoded from Yahoo Finance.  I used only closing prices to try and predict , next
  prices. Every 19 elements are taken into prediction, and we tried to predict next price. 
  
  We used 70% of data set for training , and 30% for testing purposes.
