# This module helps to get size of input image required by each face recognition model user selects

def find_input_shape(model):

	input_shape = model.layers[0].input_shape

	if type(input_shape) == list:
		input_shape = input_shape[0][1:3]
	else:
		input_shape = input_shape[1:3]


	if type(input_shape) == list: 
		input_shape = tuple(input_shape)

	return input_shape