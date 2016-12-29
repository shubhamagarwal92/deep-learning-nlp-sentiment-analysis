"""
Based on Xiang Zhang, Junbo Zhao, Yann LeCun. Character-level Convolutional Networks for Text Classification. 
Advances in Neural Information Processing Systems 28 (NIPS 2015)
https://github.com/zhangxiangxiao/Crepe

Inspired by https://github.com/johnb30/py_crepe

Another good explanation can be found at 
https://gab41.lab41.org/deep-learning-sentiment-one-character-at-a-t-i-m-e-6cd96e4f780d#.ie3szwhi2
"""

nb_filter = 256
#Number of units in the dense layer
# dense_outputs = 1024
#Conv layer kernel size
filter_kernels = [7, 7, 3, 3, 3, 3]
#Number of units in the final output layer. Number of classes.
# cat_output = 4
#Compile/fit params
batch_size = 80
nb_epoch = 10
from keras.layers import Input, Flatten
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, Convolution1D, MaxPooling1D
maxlen = 1014
vocab_size = 67
fully_connected = [1024,1024,1]
model = Sequential()
model.add(Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[0],
                         border_mode='valid', activation='relu',
                         input_shape=(maxlen, vocab_size)))
model.add(MaxPooling1D(pool_length=3))
model.add(Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[1],
                          border_mode='valid', activation='relu'))
model.add(MaxPooling1D(pool_length=3))
model.add(Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[2],
                          border_mode='valid', activation='relu'))
model.add(Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[3],
                          border_mode='valid', activation='relu'))
model.add(Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[4],
                          border_mode='valid', activation='relu'))
model.add(Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[5],
                          border_mode='valid', activation='relu'))
model.add(MaxPooling1D(pool_length=3))
model.add(Flatten())
model.add(Dense(fully_connected[0]))
model.add(Dropout(0.5))
model.add(Activation('relu'))
#Input is 1024 Output is 1024
model.add(Dense(fully_connected[1]))
model.add(Dropout(0.5))
model.add(Activation('relu'))
#Input is 1024 Output is 1
model.add(Dense(fully_connected[2]))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


model.fit(npa, sentiment_numpy, batch_size=batch_size, nb_epoch=4)

print('Train...')


model.fit(npa, sentiment_numpy, batch_size=batch_size, nb_epoch=4)
