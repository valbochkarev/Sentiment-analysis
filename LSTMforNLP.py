# -*- coding: utf-8 -*-
"""
Created on Tue May 28 02:08:18 2019

@author: ValfkNote
"""

max_features = 100000
# In[2]: proccesing dataset
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
# define the document

# download data 
data = pd.read_csv("preprocessTrain.csv", delimiter=",",encoding = "latin-1",index_col=False) #or encoding = "latin-1"
data = shuffle(data)
data = data[:300000]
data.columns = ['id', 'sentiment', 'text']
data.drop(['id'], axis=1)
print(len(data[ data['sentiment'] == 1]))
print(len(data[ data['sentiment'] == 0]))
print(len(data))


# In[3]: Creating train data
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import text_to_word_sequence as TtW_sequence
def list_creator(data):
    a = list()
    for el in data:
        a.append(list(TtW_sequence(el)))
    return a

maxlen = 50

X = list_creator(data['text'])
X = pad_sequences(X, maxlen=maxlen)


Y = pd.get_dummies(data['sentiment']).values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3, random_state = 42)
X_validate, X_test, Y_validate, Y_test = train_test_split(X_test,Y_test, test_size = 0.8, random_state = 13)

print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)


# In[4]: creating model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.callbacks import TensorBoard  

tensorboard=TensorBoard(log_dir='./logs', write_graph=True)

    
batch_size = 128
embed_dim = 128
max_features = 100000
model = Sequential()
model.add(Embedding(max_features,  output_dim = embed_dim, input_length=X_train.shape[1]))
model.add(LSTM(embed_dim, return_sequences=True))
model.add(LSTM(embed_dim))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# In[5]: Train neural network
model.fit(X_train, Y_train, epochs = 2, batch_size=batch_size, validation_data=(X_test, Y_test), verbose = 2)
score,acc = model.evaluate(X_validate, Y_validate, verbose = 2, batch_size = batch_size)
print("score: %.2f" % (score))
print("acc: %.2f" % (acc))
pos_cnt, neg_cnt, pos_correct, neg_correct = 0, 0, 0, 0
for x in range(len(X_validate)):
    
    result = model.predict(X_validate[x].reshape(1,X_validate.shape[1]),batch_size=1,verbose = 2)[0]
   
    if np.argmax(result) == np.argmax(Y_validate[x]):
        
        if np.argmax(Y_validate[x]) == 0:
            neg_correct += 1
        else:
            pos_correct += 1
       
    if np.argmax(Y_validate[x]) == 0:
        neg_cnt += 1
    else:
        pos_cnt += 1



print("pos_acc", pos_correct/pos_cnt*100, "%")
print("neg_acc", neg_correct/neg_cnt*100, "%")

# In[6]: Save model
model_json = model.to_json()
json_file = open("LSTM_model.json", "w")
json_file.write(model_json)
json_file.close()
model.save_weights("LSTM_model.h5")