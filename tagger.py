#!/usr/bin/env python

import glob
import json
import os
import time
import codecs
import optparse
import numpy as np
from tqdm import tqdm

from nltk.tokenize import sent_tokenize, word_tokenize

from loader import prepare_sentence, format_input_for_tagging, format_filename
from utils import create_input, iobes_iob, zero_digits, avg, softmax
from model import Model
from config import prediction_parameters as parameter

# Check parameters validity
assert os.path.isdir(parameter["MODEL"])
assert os.path.isdir(parameter["INPUT"])


def get_research_methods(triples):
    """
    Get chunks from IOB tags of research methods
    Input: list of triple [word, tag, score] of the paper got from tagger
    Ouput: list of research methods with scores
    """
    tags = ['O', 'B-RS', 'I-RS']
    top = 'O'
    stack_rs = []
    stack_sc = []
    dict_research_method = {}

    # stack to get the research methods from paper
    for i in range(len(triples)):
        if triples[i][1] == 'B-RS':
            if top == 'O':
                stack_rs.append(triples[i][0])
                stack_sc.append(softmax(triples[i][2])[tags.index(triples[i][1])])
                top = 'B-RS'
            else:
                research_method = ' '.join(stack_rs)
                dict_research_method[research_method] = avg(stack_sc)
                stack_rs = [triples[i][0]]
                stack_sc = [triples[i][2]]
                top = 'B-RS'
                
        elif triples[i][1] == 'I-RS':
            if top == 'O':
                continue
            else:
                stack_rs.append(triples[i][0])
                stack_sc.append(softmax(triples[i][2])[tags.index(triples[i][1])])
                top = 'I-RS'
        
        else:
            if top == 'O':
                continue
            else:
                research_method = ' '.join(stack_rs)
                dict_research_method[research_method] = avg(stack_sc)                
                stack_rs = []
                stack_sc = []
                top = 'O'
        
    return dict_research_method

# Load existing model
print "Loading model..."
model = Model(model_path=parameter["MODEL"])
parameters = model.parameters

# Load reverse mappings
word_to_id, char_to_id, tag_to_id = [
    {v: k for k, v in x.items()}
    for x in [model.id_to_word, model.id_to_char, model.id_to_tag]
]

# Load the model
_, f_eval = model.build(training=False, **parameters)
model.reload()

results = []

print 'Tagging...'
#with codecs.open(opts.input, 'r', 'utf-8') as f_input:
with open("%s" %parameter["JSON_FILE"], "r") as json_file:
    filenames = json.load(json_file)
    list_input_file = [doc["text_file_name"] for doc in filenames]


for filename in list_input_file:
    triples = []
    list_research_method = []
    try:
        with codecs.open("%s%s" %(parameter["INPUT"], filename), 'r', 'utf-8') as f_input:
            sentences = format_input_for_tagging(f_input.read())
            

            for line in tqdm(sentences):
                words = line.rstrip().split()

                if line:
                    # Lowercase sentence
                    if parameters['lower']:
                        line = line.lower()
                    # Replace all digits with zeros
                    if parameters['zeros']:
                        line = zero_digits(line)
                    # Prepare input
                    sentence = prepare_sentence(words, word_to_id, char_to_id,
                                                lower=parameters['lower'])
                    input = create_input(sentence, parameters, False)
                    # Decoding
                    if parameters['crf']:
                        #y_preds, score = np.array(f_eval(*input))[1:-1]
                        scores = f_eval(*input)[1]
                        y_preds= np.array(f_eval(*input)[0])[1:-1]
                    else:
                        y_preds = f_eval(*input).argmax(axis=1)
                        scores = f_eval(*input)   
                    y_preds = [model.id_to_tag[y_pred] for y_pred in y_preds]
                    # Output tags in the IOB2 format
                    if parameters['tag_scheme'] == 'iobes':
                        y_preds = iobes_iob(y_preds)
                    # Write tags
                    assert len(y_preds) == len(words)

                    for word, tag, score in zip(words, y_preds, scores):
                        triples.append([word, tag, score])  
                else:
                    continue
        list_research_method = get_research_methods(triples)

        for pair in list_research_method:
            tmp_res = dict({
                    "publication_id": format_filename(filename[-9:-4]),
                    "method": pair,
                    "score": float("%.3f" %list_research_method[pair])
                    })
            if tmp_res["score"] >= parameter["THRESHOLD"]:
                results.append(tmp_res)
    except:
        continue

with codecs.open(parameter["OUTPUT"], 'w', 'utf-8') as f_output:
    json.dump(results, f_output)