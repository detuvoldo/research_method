#!/usr/bin/env python

import os
import numpy as np
import itertools
from collections import OrderedDict
from utils import create_input
import loader
import time

from utils import models_path, evaluate, eval_script, eval_temp
from loader import word_mapping, char_mapping, tag_mapping
from loader import update_tag_scheme, prepare_dataset
from loader import augment_with_pretrained
from model import Model
from config import training_parameters, training_path,reload

start_time = int(time.time())

# Check training_parameters validity
assert os.path.isfile(training_path["TRAIN"])
assert os.path.isfile(training_path["DEV"])
assert os.path.isfile(training_path["TEST"])
assert training_parameters['char_dim'] > 0 or training_parameters['word_dim'] > 0
assert 0. <= training_parameters['dropout'] < 1.0
assert training_parameters['tag_scheme'] in ['iob', 'iobes']
assert not training_parameters['all_emb'] or training_parameters['pre_emb']
assert not training_parameters['pre_emb'] or training_parameters['word_dim'] > 0
assert not training_parameters['pre_emb'] or os.path.isfile(training_parameters['pre_emb'])

# Check evaluation script / folders
if not os.path.isfile(eval_script):
    raise Exception('CoNLL evaluation script not found at "%s"' % eval_script)
if not os.path.exists(eval_temp):
    os.makedirs(eval_temp)
if not os.path.exists(models_path):
    os.makedirs(models_path)

# Initialize model
model = Model(parameters=training_parameters, models_path=models_path)
print "Model location: %s" % model.model_path

# Data training_parameters
lower = training_parameters['lower']
zeros = training_parameters['zeros']
tag_scheme = training_parameters['tag_scheme']

# Load sentences
train_sentences = loader.load_sentences(training_path["TRAIN"], lower, zeros)
dev_sentences = loader.load_sentences(training_path["DEV"], lower, zeros)
test_sentences = loader.load_sentences(training_path["TEST"], lower, zeros)

# Use selected tagging scheme (IOB / IOBES)
update_tag_scheme(train_sentences, tag_scheme)
update_tag_scheme(dev_sentences, tag_scheme)
update_tag_scheme(test_sentences, tag_scheme)

# Create a dictionary / mapping of words
# If we use pretrained embeddings, we add them to the dictionary.
if training_parameters['pre_emb']:
    dico_words_train = word_mapping(train_sentences, lower)[0]
    dico_words, word_to_id, id_to_word = augment_with_pretrained(
        dico_words_train.copy(),
        training_parameters['pre_emb'],
        list(itertools.chain.from_iterable(
            [[w[0] for w in s] for s in dev_sentences + test_sentences])
        ) if not training_parameters['all_emb'] else None
    )
else:
    dico_words, word_to_id, id_to_word = word_mapping(train_sentences, lower)
    dico_words_train = dico_words

# Create a dictionary and a mapping for words / POS tags / tags
dico_chars, char_to_id, id_to_char = char_mapping(train_sentences)
dico_tags, tag_to_id, id_to_tag = tag_mapping(train_sentences)

# Index data
train_data = prepare_dataset(
    train_sentences, word_to_id, char_to_id, tag_to_id, lower
)
dev_data = prepare_dataset(
    dev_sentences, word_to_id, char_to_id, tag_to_id, lower
)
test_data = prepare_dataset(
    test_sentences, word_to_id, char_to_id, tag_to_id, lower
)

print "%i / %i / %i sentences in train / dev / test." % (
    len(train_data), len(dev_data), len(test_data))

# Save the mappings to disk
print 'Saving the mappings to disk...'
model.save_mappings(id_to_word, id_to_char, id_to_tag)

# Build the model
f_train, f_eval = model.build(**training_parameters)

# Reload previous model values
if reload:
    print 'Reloading previous model...'
    model.reload()

#
# Train network
#
singletons = set([word_to_id[k] for k, v
                  in dico_words_train.items() if v == 1])
n_epochs = 50  # number of epochs over the training set
freq_eval = 1000  # evaluate on dev every freq_eval steps
best_dev = -np.inf
best_test = -np.inf
count = 0
#trainLog = open('train.log', 'w')
for epoch in xrange(n_epochs):
    epoch_costs = []
    print "Starting epoch %i..." % epoch
    for i, index in enumerate(np.random.permutation(len(train_data))):
        count += 1
        input = create_input(train_data[index], training_parameters, True, singletons)
        new_cost = f_train(*input)
        epoch_costs.append(new_cost)
        if i % 50 == 0 and i > 0 == 0:
            print "%i, cost average: %f" % (i, np.mean(epoch_costs[-50:]))
        if count % freq_eval == 0:
            dev_score = evaluate(training_parameters, f_eval, dev_sentences,
                                 dev_data, id_to_tag, dico_tags, epoch)
            test_score = evaluate(training_parameters, f_eval, test_sentences,
                                  test_data, id_to_tag, dico_tags, epoch)
            print "Score on dev: %.5f" % dev_score
            print "Score on test: %.5f" % test_score
            if dev_score > best_dev:
                best_dev = dev_score
                best_dev_epoch = epoch
                #print "New best score on dev at epoch %i." % best_epoch
                #model.save()
            if test_score > best_test:
                best_test = test_score
                best_test_epoch = epoch
                print "Saving model to disk..."
                model.save()
                #print "New best score on test."
            print "Best score on dev at epoch %i." % best_dev_epoch
            print "Best score on test at epoch %i." % best_test_epoch
    print "Epoch %i done. Average cost: %f" % (epoch, np.mean(epoch_costs))

print(time.time() - start_time)

with open('./evaluation/temp/train.log', 'w') as trainLog:
    trainLog.write('Best score on dev at epoch: %i\n' % best_dev_epoch)
    trainLog.write('Best score on test at epoch: %i\n\n' % best_test_epoch)
    
    with open('./evaluation/temp/eval.%i.scores' % best_dev_epoch) as best_dev_res:
        trainLog.write('Best result on dev set at epoch %i:\n' % best_dev_epoch)
        for line in best_dev_res:
            trainLog.write('%s' % line)

    with open('./evaluation/temp/eval.%i.scores' % best_test_epoch) as best_test_res:
        trainLog.write('\nBest result on test set at epoch %i:\n' % best_test_epoch)
        for line in best_test_res:
            trainLog.write('%s' % line)
