import sys
import os

import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

# import theano
# print theano.sandbox.cuda.dnn.dnn_available()


######### we can improve the performance by removing a 
######### few characters from the dictionary



########### How to form the training data 
########### 1. use a fixed length, like 100 characters
########### 2. use complete sentences, pad shorter ones and truncate longer.


###### input to LSTMs: [nb_samples,timesteps,input_dim]
###### nb_samples: number of rows of data X
###### timesteps: sequence length
###### input dim: dimension of the input, here its just 1,ie for every node at t, dimension is 1.
###### output of LSTMs: [nb_samples,timesteps,output_dim] if return_sequences else [nb_samples,output_dim]



###### LSTM later with 256 memory units probably means the hidden state has dimension of 256.
###### the model comprises of an LSTM layer, followed by a dropout layer and a dense layer.
###### check point callback function: https://keras.io/callbacks/, http://machinelearningmastery.com/check-point-deep-learning-models-keras/


####### Steps to follow on AWS ########
####### 1. SSH into the AMI and then scp the files. #######
####### 2. Check if Keras and Theano are working preoperly on the AMI #######
####### 3. Pricing - $0.702 per Hour


def get_loss(predictions_for_targets):
	N = len(predictions_for_targets)
	l = -1 * numpy.sum(numpy.log(predictions_for_targets))
	return l/float(N)

####### Testing part starts here #########
def generate_text(model,pattern,target_text,n_vocab,char_to_int,checkpoint_file):
	# model.load_weights(checkpoint_file)
	predictions_for_targets = []
	for i in range(1000):
		x = numpy.reshape(pattern,(1,seq_length,1))
		x = x/float(n_vocab)
		y = target_text[i]
		target_index = char_to_int[y]
		prediction = model.predict(x,verbose=0)
		index = numpy.argmax(prediction)
		prediction_for_target = prediction[target_index]
		predictions_for_targets.append(prediction_for_target)
		result  = int_to_char[index]
		sys.stdout.write(result)
		pattern.append(index)
		pattern = pattern[1:len(pattern)]

	return predictions_for_targets


def start_training(model,checkpoint_file,nb_epoch=50):
	# filepath = "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
	checkpoint = ModelCheckpoint(checkpoint_file,monitor='loss',verbose=1,save_best_only=True,mode='min')
	callbacks_list = [checkpoint]
	model.fit(X,y,nb_epoch=nb_epoch,batch_size=128,callbacks=callbacks_list)
	return model


def define_model():
	model = Sequential()
	model.add(LSTM(256,input_shape=(X.shape[1],X.shape[2])))
	model.add(Dropout(0.2))
	##### added a layer ######
	model.add(LSTM(256))
	model.add(Dropout(0.2))
	##########################
	model.add(Dense(y.shape[1],activation='softmax'))
	model.compile(loss='categorical_crossentropy',optimizer='adam')
	return model


def get_test_data(raw_text,char_to_int):
	raw_text = [c if c in char_to_int else "UNKNOWN_TOKEN" for c in raw_text]
	starting_pattern = [char_to_int[c] for c in raw_text_test[:seq_length]]
	print "Starting pattern ....\n" + ''.join([c for c in raw_text_test[:seq_length]]) 
	target_text = [char_to_int[c] for c in raw_text_test[seq_length:]]

	return (starting_pattern,target_text)



def get_training_data(raw_text,n_vocab,n_chars,char_to_int,seq_length=100):
	dataXTraining = []
	dataYTraining = []
	for i in range(0,n_chars-seq_length,1):
		seq_in = raw_text[i:i+seq_length]
		seq_out = raw_text[i+seq_length]
		dataXTraining.append([char_to_int[char] for char in seq_in])
		dataYTraining.append(char_to_int[seq_out])

	n_pattern = len(dataXTraining)
	print "Total patterns: ",n_pattern

	X = numpy.reshape(dataXTraining,(n_pattern,seq_length,1))
	X = X/float(n_vocab)
	########### We have to add nb_classes to the function since 
	########### the data will be there in training (everything is seen)
	########### but not in test.
	y = np_utils.to_categorical(dataYTraining,nb_classes=50)

	# print len(X[0])
	# print y[0]
	# print dataYTraining[0]
	# exit(0)

	return (X,y)


def get_dictionary(raw_text):
	# unknown_token = "UNKNOWN_TOKEN"
	
	chars = sorted(list(set(raw_text)))
	char_to_int = dict((c,i) for i,c in enumerate(chars))
	# char_to_int[unknown_token] = len(char_to_int.keys())
	int_to_char = dict((i,c) for i,c in enumerate(chars))
	# int_to_char[len(int_to_char.keys())] = unknown_token
	
	n_vocab = len(char_to_int.keys())
	n_chars = len(raw_text)

	print "Character to Int: ",char_to_int
	print "Int to Character: ", int_to_char
	print 'Length of vocabulary: ',n_vocab
	
	###### checking #########
	# freq_dict = {}
	# for c in raw_text:
	# 	if c in ['\x80','\x99','\x98','\x9d','\x9c']:
	# 		print c
	# 	if c in freq_dict:
	# 		freq_dict[c] += 1
	# 	else:
	# 		freq_dict[c] = 1
	# print freq_dict
	# exit(0)
	#########################
	return (char_to_int,int_to_char,n_vocab,n_chars)

######## we can tokenize text as well ########
######## replace the tokens with space #######
def do_preprocessing(fname):
	unknown_token = "UNKNOWN_TOKEN"
	raw_text = open(fname).read()
	raw_text = raw_text.lower()
	########## Based on their frequency and relevance in the text ############
	raw_text = [c if c not in ['\xe2','\x80','\x99','\x98','\x9d','\x9c','0','3','*','[',']','_'] \
		else unknown_token for c in raw_text]

	return raw_text

def check_file_exists(model):
	filepath = os.path.expanduser('~') + '/text-generation/weights-best.hdf5'
	try:
		model.load_weights(filepath)
		print "Weights loaded!"
		return (True,model)
	except Exception, e:
		print e
		return (False,model)

if __name__ == '__main__':
	file_training = "alice-in-wonderland-training.txt"
	filename_test = "alice-in-wonderland-test.txt"
	checkpoint_file = "weights-best.hdf5"
	seq_length = 100

	print "\n\nPre-process the training data ....\n\n"
	raw_text_training = do_preprocessing(file_training)
	print "\n\nGetting the vocabulary for the language model ....\n\n"
	char_to_int,int_to_char,n_vocab,n_chars = get_dictionary(raw_text_training)
	print "\n\nGetting the training data ....\n\n"
	X,y = get_training_data(raw_text_training,n_vocab,n_chars,char_to_int,seq_length)
	print "\n\nPre-process the test data ....\n\n"
	raw_text_test = do_preprocessing(filename_test)
	print "\n\nGetting the test data ....\n\n"
	starting_pattern,target_text = get_test_data(raw_text_test,char_to_int)
	model = define_model()
	flag,model = check_file_exists(model)
	if flag:
		model = start_training(model,checkpoint_file,50) ##### change epochs here.
	else:
		model = start_training(model,checkpoint_file)
	predictions_for_targets = generate_text(model,starting_pattern,target_text,n_vocab,char_to_int,checkpoint_file)
	loss= get_loss(predictions_for_targets)

	print "Loss on generated text is: ",loss



