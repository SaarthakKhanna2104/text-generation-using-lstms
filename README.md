# text-generation-using-lstms

1. Project Overview

This project is about generating text by training our LSTM model on a training corpus. The training data being used by us is the "Alice in Wonderland" book. The trained LSTM model will generate text similar to the one in Alice in Wonderland.

This project consists of:

a) Getting the data and dividing it into 2 parts. The first part included 11 chapters from the book and were used for training ours LSTMs and the second part included 1 chapter which will be used as a test set for evaluating our LSTMs.
b) After that came the preprocessing phase. The preprocessing phase included 
	i) converting the text to lower case in order to reduce the sixe of the token dictionary.
	ii) removing the tokens which have very low frequency of occurance since they won't contribute much in the learning 	phase of the LSTM. This is how the frequencies of the tokens were distributed:

		{'\x80': 2752, '\n': 3051, '\x99': 1619, '\x98': 1030, '\x9d': 47, '\x9c': 56, '!': 432, ' ': 23061, ')': 47, '(': 42, '*': 60, '-': 599, ',': 2215, '.': 903, '0': 1, '3': 1, ';': 181, ':': 217, '?': 191, 'b': 1339, '[': 7, ']': 2, '_': 2, 'a': 8123, 'c': 2230, '\xe2': 2752, 'e': 12400, 'd': 4531, 'g': 2326, 'f': 1833, 'i': 6891, 'h': 6741, 'k': 1057, 'j': 117, 'm': 1928, 'l': 4373, 'o': 7570, 'n': 6425, 'q': 191, 'p': 1392, 's': 5986, 'r': 4971, 'u': 3188, 't': 9817, 'w': 2490, 'v': 762, 'y': 2097, 'x': 136, 'z': 68}

c) Converting the training data into a form which could be used as an input to LSTM. We have split the text into subsequences of ”SEQUENCE_LENGTH” which is variable. The training pattern consists of SEQUENCE_LENGTH time steps of X and a character output y. For example, if SEQUENCE_LENGTH = 5, then the input data will look like: CHAPT -> E, HAPTE -> R. These characters are converted into integers understood by a Neural Network by using a character to integer dictionary. The training data had approximately 15,000 instances.

d) Building the neural network architecture. The architecture I was working on had:
	i) LSTM layer with 256 memory units.
	ii) Dropout layer with dropout parameter = 0.2.
	iii) Dense layer with activation function = softmax.

e) Training the model. The Neural Network was trained using gradient descent. I ran it for 10 epochs and 50 epochs with a batch size of 128. While training, I keep saving the weights of the model with the minimum loss. The loss function used is categorical cross-entropy loss function.

f) Evaluating the model. The model evaluation was done by providing the our trained LSTM model with the first SEQUENCE_LENGTH tokens from the test set. The predicted output was compared with the actual output and the loss was calculated using the categorical cross-entropy loss function.

This is how the output for the first 1000 tokens looked after:

	i) 10 epochs: ke  and the was ao anl she whit was a little to ae and the was ao anl aaai to the wabten  and the was ao anl the whit was aol the was ao anl aaai to the wabten  and the was ao anl the whit was aol the was ao anl aaak to the wabten  and the was ao anl the whit was aol the was ao anl aaak to the wabten  and the was ao anl the whit was aol the was ao anl aaak to the wabten  and the was ao anl the whit was aol the was ao anl aaak to the wabten  and the was ao anl the whit was aol the was ao anl aaak to the wabten  and the was ao anl the whit was aol the was ao anl aaak to the wabten  and the was ao anl the whit was aol the was ao anl aaak to the wabten  and the was ao anl the whit was aol the was ao anl aaak to the wabten  and the was ao anl the whit was aol the was ao anl aaak to the wabten  and the was ao anl the whit was aol the was ao anl aaak to the wabten  and the was ao anl the whit was aol the was ao anl aaak to the wabten  and the was ao anl the whit was aol the was ao anl aaak to

	Loss = 4.0807565918

	ii) 50 epochs: ked to the was oot and the was to the was oot and the was to the was to the was to the was to the was to the was to the was to the was oot and the was to the was to the was to the was to the was to the was to the was to the was oot and the was to the was to the was to the was to the was to the was to the was to the was oot and the was to the was to the was to the was to the was to the was to the was to the was oot and the was to the was to the was to the was to the was to the was to the was to the was oot and the was to the was to the was to the was to the was to the was to the was to the was oot and the was to the was to the was to the was to the was to the was to the was to the was oot and the was to the was to the was to the was to the was to the was to the was to the was oot and the was to the was to the was to the was to the was to the was to the was to the was oot and the was to the was to the was to the was to the was to the was to the was to the was oot and the was to the was t

	Loss = 3.25673364258

