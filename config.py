# config.py

from collections import OrderedDict

prediction_parameters = dict({
	"INPUT": "./test_data/", # path to data folder
	"OUTPUT": "res.txt", # path to output folder
	"MODEL": "./models/bi-lstm-crf/", # path to prediction model
	"JSON_FILE": "./publications.json",
	"THRESHOLD": 2.0/3
	})

training_path = dict({
	"TRAIN": "train.txt",
	"DEV": "dev.txt",
	"TEST": "test.txt" 
	})

training_parameters = OrderedDict()

training_parameters['tag_scheme'] = "iob"
training_parameters['lower'] = False
training_parameters['zeros'] = False
training_parameters['char_dim'] = 25
training_parameters['char_lstm_dim'] = 25
training_parameters['char_bidirect'] = False
training_parameters['word_dim'] = 100
training_parameters['word_lstm_dim'] = 100
training_parameters['word_bidirect'] = True
training_parameters['pre_emb'] = "skip100"
training_parameters['all_emb'] = False
training_parameters['cap_dim'] = 0
training_parameters['crf'] = True
training_parameters['dropout'] = 0.5
training_parameters['lr_method'] = "sgd-lr_.005"

reload = False