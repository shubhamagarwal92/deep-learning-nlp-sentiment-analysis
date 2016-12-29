


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation

max_features = 68
embedding_dims = 32
# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model = Sequential()
model.add(Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen,
                    dropout=0.2))
model.add(LSTM(embedding_dims, dropout_W=0.2, dropout_U=0.2))  # try using a GRU instead, for fun
model.add(Dense(1))
model.add(Activation('relu'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
batch_size =32
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=5)
          # validation_data=(X_test, y_test))
score, acc = model.evaluate(X_test, y_test,
                            batch_size=batch_size)


# Epoch 1/5
# 25000/25000 [==============================] - 204s - loss: 0.7605 - acc: 0.5069     
# Epoch 2/5
# 25000/25000 [==============================] - 209s - loss: 0.6904 - acc: 0.5283     
# Epoch 3/5
# 25000/25000 [==============================] - 215s - loss: 0.6875 - acc: 0.5334     
# Epoch 4/5
# 25000/25000 [==============================] - 210s - loss: 0.6840 - acc: 0.5462     
# Epoch 5/5
# 25000/25000 [==============================] - 211s - loss: 0.6790 - acc: 0.5560     
