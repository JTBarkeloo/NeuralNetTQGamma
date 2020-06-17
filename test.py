from keras.models import Model,Sequential
from keras.layers import Dense, Dropout, Input

def DNNmodel(Input_shape=(10,), n_hidden=1, n_nodesHidden=10, dropout=0.2, optimizer='adam'):
	inputs=Input(shape=Input_shape)
	i=0
	if n_hidden>0:
		hidden=Dense(n_nodesHidden, activation='relu')(inputs)
	        hidden=Dropout(dropout)(hidden)
		i+=1
	while i<n_hidden:
		hidden=Dense(n_nodesHidden, activation='relu')(hidden)
                hidden=Dropout(dropout)(hidden)
		i+=1
	outputs = Dense(1,activation='sigmoid')(hidden)
	model = Model(inputs,outputs)
	model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy']) 
	model.summary()
	return model
inshape=(16,)
model=DNNmodel()
