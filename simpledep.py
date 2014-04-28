#!/usr/bin/env python3

from __future__ import print_function, unicode_literals

'''
Original License from http://people.ict.usc.edu/~sagae/parser/simpledep.pl
(downloaded from http://people.ict.usc.edu/~sagae/software.html):

Copyright (c) 2011, Kenji Sagae
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

'''
A very simple shift-reduce parser.
Many people don't realize the Yamada & Matsumoto 2003
dependency parsing algorithm
differs from arc-standard shift-reduce
dependency parsing by just one line of code.
This is an arc-standard parser, implemented
in the style of Yamada & Matsumoto.

-- Kenji Sagae

This was adapted from perl to python in 2014.
Note that it does not perform exactly the same as the original perl
script, probably due to different ways of
seeding random number generators and/or breaking ties.

-- Michael Heilman, mheilman@ets.org

'''

import argparse
import json
import random
import logging
from collections import defaultdict


LEFTWALL = {'idx': 0,
            'word': "LeftWall",
            'lemma': "leftwall",
            'cpos':  "lw",
            'pos': "LW",
            'morph': "_",
            'glink': 0,
            'glabel': "_",
            'link': 0,
            'label': "_",
            'ch': [],
            'lch': [],
            'rch': []}

RIGHTWALL = {'idx': 0,
             'word': "RightWall",
             'lemma': "rightwall",
             'cpos': "rw",
             'pos': "RW",
             'morph': "_",
             'glink': 0,
             'glabel': "_",
             'link': 0,
             'label': "_",
             'ch': [],
             'lch': [],
             'rch': []}

DUMMY_WORD = {'idx': 0,
              'word': "NONE",
              'lemma': "none",
              'cpos': "none",
              'pos': "none",
              'morph': "_",
              'glink': 0,
              'glabel': "_",
              'link': 0,
              'label': "_",
              'ch': [],
              'lch': [],
              'rch': []}


def classify(wdep, features, allow_shift=True):
    '''
    Classify one instance according to our linear classification model.

    :param features: set of features for which we need to select an action
    :returns: the label for the best action to take
    '''

    bestlabel = None
    bestscore = float('-inf')

    for a in wdep['**ALL ACTS**']:
        if not allow_shift and a == "S":
            continue
        wdep_a = wdep[a]
        score = 0.0
        for f in features:
            if f in wdep_a:
                score += wdep_a[f]
        if score > bestscore:
            bestscore = score
            bestlabel = a

    return bestlabel


def print_conll(s):
    '''
    print one sentence (dependency tree) in CoNLL format
    '''

    keys = ['idx', 'word', 'lemma', 'cpos', 'pos', 'morph', 'link', 'label']
    for i in range(1, len(s)):
        print('\t'.join('{}'.format(s[i][key]) for key in keys))
    print()


def read_sentence_conll(fp):
    '''
    read one sentence (dependency tree) in CoNLL format from STDIN

    :param fp: a file pointer for the input
    :return: a list of dict objects, one for each word
    '''

    # skip blank lines at the start of the input
    input_str = fp.readline()
    while input_str and input_str == '\n':
        input_str = fp.readline()

    # return if there are no nonblank lines
    if not input_str:
        # TODO raise exception instead?
        return []

    # start with a copy of the dummy leftwall node
    sent = [dict(LEFTWALL)]

    while input_str:
        parts = input_str.split()
        if len(parts) < 3:
            break

        tok = {'idx': int(parts[0]),
               'word': parts[1],
               'lemma': parts[2],
               'cpos': parts[3],
               'pos': parts[4],
               'morph': parts[5],
               'glink': int(parts[6]),
               'glabel': parts[7],
               'link': 0,
               'label': "_",
               'ch': [],
               'lch': [],
               'rch': []}
        sent.append(tok)
        input_str = fp.readline()

    if len(sent) <= 1:
        # only the left wall is included, so return empty list
        # TODO raise exception?
        return []

    return sent


def make_defaultdict():
    '''
    utility function for making nested defaultdict objects
    (for feature_name->parse_action->value dictionaries for weights)
    '''
    return defaultdict(float)


def mkfeats(i, prevact, s):
    '''
    assemble an array of features from the current parser state
    '''

    # initialize the previous and
    # next words to be dummy words,
    # in case there are no previous
    # or next words
    prev2 = dict(LEFTWALL)
    # TODO is it necessary to remove ch, lch, rch from prev2?
    prev1 = dict(LEFTWALL)
    next1 = dict(RIGHTWALL)
    next2 = dict(RIGHTWALL)
    next3 = dict(RIGHTWALL)

    # lw is the second item on the stack
    lw = s[i]

    # rw is the item on top of the stack
    rw = s[i + 1]

    # now get the actual previous and next words
    # if they exist
    if i > 1:
        prev1 = s[i - 1]
    if i > 2:
        prev2 = s[i - 2]

    sent_len = len(s)
    if i < sent_len - 2:
        next1 = s[i + 2]
    if i < sent_len - 3:
        next2 = s[i + 3]
    if i < sent_len - 4:
        next3 = s[i + 4]

    # llch is the leftmost child of lw so far
    llch = lw['lch'][0] if lw['lch'] else dict(DUMMY_WORD)

    # lrch is the rightmost child of lw so far
    lrch = lw['rch'][0] if lw['rch'] else dict(DUMMY_WORD)

    # rlch is the leftmost child of rw so far
    rlch = rw['lch'][0] if rw['lch'] else dict(DUMMY_WORD)

    # rrch is the rightmost child of rw so far
    rrch = rw['rch'][0] if rw['rch'] else dict(DUMMY_WORD)

    # get the number of left and right
    # children for lw and rw
    lnch = len(lw['ch'])
    llnch = len(lw['lch'])
    lrnch = len(lw['rch'])

    rnch = len(rw['ch'])
    rlnch = len(rw['lch'])
    rrnch = len(rw['rch'])

    # now add all the features to a Counter (dict subtype) object
    feats = []

    feats.append('**BIAS**')

    # first the word, POS tag and dependency label
    # for lw
    feats.append('lw:{}'.format(lw['word']))
    feats.append('lwpos:{}'.format(lw['pos']))
    feats.append('lwlab:{}'.format(lw['label']))

    # same for rw
    feats.append('rw:{}'.format(rw['word']))
    feats.append('rwpos:{}'.format(rw['pos']))
    feats.append('rwlab:{}'.format(rw['label']))

    # then info about previous and next words
    # (previous words are on the stack,
    # next words are in the input list)
    feats.append('p1:{}'.format(prev1['word']))
    feats.append('p1pos:{}'.format(prev1['pos']))
    feats.append('p1label:{}'.format(prev1['label']))

    feats.append('p2:{}'.format(prev2['word']))
    feats.append('p2pos:{}'.format(prev2['pos']))

    feats.append('n1:{}'.format(next1['word']))
    feats.append('n1pos:{}'.format(next1['pos']))

    feats.append('n2:{}'.format(next2['word']))
    feats.append('n2pos:{}'.format(next2['pos']))

    feats.append('n3pos:{}'.format(next3['pos']))

    # the info about the children of lw and rw

    feats.append('llchw:{}'.format(llch['word']))
    feats.append('llchpos:{}'.format(llch['pos']))
    feats.append('llchlab:{}'.format(llch['label']))
    feats.append('lrchw:{}'.format(lrch['word']))
    feats.append('lrchpos:{}'.format(lrch['pos']))
    feats.append('lrchlab:{}'.format(lrch['label']))

    feats.append('rlchw:{}'.format(rlch['word']))
    feats.append('rlchpos:{}'.format(rlch['pos']))
    feats.append('rlchlab:{}'.format(rlch['label']))
    feats.append('rrchw:{}'.format(rrch['word']))
    feats.append('rrchpos:{}'.format(rrch['pos']))
    feats.append('rrchlab:{}'.format(rrch['label']))

    feats.append('lnch:{}'.format(lnch))
    feats.append('llnch:{}'.format(llnch))
    feats.append('lrnch:{}'.format(lrnch))

    feats.append('rnch:{}'.format(rnch))
    feats.append('rlnch:{}'.format(rlnch))
    feats.append('rrnch:{}'.format(rrnch))

    # some feature combinations

    # the following loop improves accuracy by
    # quite a bit, but creates A LOT of features
    # (slower training and parsing, takes more
    # memory)
    # n = len(feats)
    # featkeys = set(feats.keys())
    # for feat in featkeys:
    #     feats.append('{}~lwpos:{}'.format(feat, lw['pos']))
    #     feats.append('{}~rwpos:{}'.format(feat, rw['pos']))
    #     feats.append('{}~n1pos:{}'.format(feat, next1['pos']))

    feats.append('lrnpos:{}\t{}\t{}'.format(lw['pos'], rw['pos'], next1['pos']))
    feats.append('lrppos:{}\t{}\t{}'.format(lw['pos'], rw['pos'], prev1['pos']))
    feats.append('lrnppos:{}\t{}\t{}\t{}'.format(lw['pos'], rw['pos'],
                                          next1['pos'], prev1['pos']))
    feats.append('lrnn2pos:{}\t{}\t{}\t{}'.format(lw['pos'], rw['pos'],
                                           next1['pos'], next2['pos']))

    feats.append('lrapos:{}~{}~{}'.format(lw['pos'], rw['pos'], prevact))
    feats.append('napos:{}~{}'.format(next1['pos'], prevact))
    feats.append('rnapos:{}~{}~{}'.format(rw['pos'], next1['pos'], prevact))
    feats.append('rapos:{}~{}'.format(rw['pos'], prevact))

    return feats


def parse(s, wdep, train=False, train_data=None):
    '''
    main parsing function

    :param train: the training flag.  If we are training, just print out
                  what we need to train a classifier offline.
                  If we are parsing, do the actual classification using
                  the linear model wdep
    :param s: the input sentence
    :param wdep: weights for linear model
    :param train_data: training data (for when train is True)
    '''

    logging.debug('parsing: {}'.format(' '.join([x['word'] for x in s[1:]])))

    # the final result (tree), initialized
    # to be just the input
    res = s

    # get the number of words in the input
    sentlen = len(s)

    # If we are training, first get the
    # number of children of each word.
    # We need this to order the parser
    # actions correctly.
    for i in range(0, sentlen):
        s[i]['numout'] = 0
    if train:
        for i in range(1, sentlen):
            s[s[i]['glink']]['numout'] += 1

    # initialize the previous action arbitrarily
    prevact = "S"

    # loop through all the words.
    # the 0th word is the dummy leftwall,
    # and the last word is the dummy rightwall,
    # so we don't need those in the loop
    i = 1
    while True:
        if i >= sentlen - 1:
            logging.debug("stopping because i >= sentlent - 1 (i = {}, sentlen = {})".format(i, sentlen))
            break

        if i < 0:
            i = 0

        if i == 0 and sentlen == 1:
            logging.debug("stopping because i == 0 and sentlen == 1")
            break

        # set lw and rw (the 2nd and top of the stack, respectively)
        lw = s[i]
        rw = s[i + 1]

        logging.debug("STACK: {}".format(' '.join([x['word'] for x in s[:i + 2]])))

        # initialize the current action to be SHIFT
        # (shift will be the default in this loop)
        act = "S"

        # get the features for the current parser state
        feats = mkfeats(i, prevact, s)

        # if we are in training mode, choose the
        # correct action, and print it out with
        # the features
        if train:
            # the action should be LEFT REDUCE
            # if lw is the head of rw,
            # and rw already has all its children
            if lw['glink'] == rw['idx'] and lw['numout'] == 0:
                act = 'L:{}'.format(lw['glabel'])

            # the action should be RIGHT REDUCE
            # if rw is the head of lw,
            # and lw already has all its children
            if rw['glink'] == lw['idx'] and rw['numout'] == 0:
                act = 'R:{}'.format(rw['glabel'])

            # add the training instance for this parse action
            train_data.append({'label': act, 'feats': feats})
            if act not in wdep['**ALL ACTS**']:
                wdep["**ALL ACTS**"][act] = 1

        # if we are parsing, just classify to get the
        # parser action
        else:
            act = classify(wdep, feats, allow_shift=(i < sentlen - 2))

        # set previous action
        prevact = act

        # now execute the action
        # (this is the same whether we are
        # training or parsing)
        if act.startswith('L:'):

            # get the dependency label from the action
            lab = act[2:]

            # count this child of rw
            # (only meaningful in training)
            rw['numout'] -= 1

            # add lw to the list of rw's children
            s[i + 1]['ch'].insert(0, s[i])
            s[i + 1]['lch'].insert(0, s[i])

            # remove lw from the stack
            s = s[:i] + s[i + 1:]

            # update the output dependency tree
            res[lw['idx']]['link'] = rw['idx']
            res[lw['idx']]['label'] = lab

            i -= 2

        # the RIGHT REDUCE action is the same, but
        # with rw and lw switched
        if act.startswith('R:'):
            lab = act[2:]
            lw['numout'] -= 1

            # add rw to the list of lw's children
            s[i]['ch'].insert(0, s[i + 1])
            s[i]['rch'].insert(0, s[i + 1])

            # remove rw from the stack
            s = s[:i + 1] + s[i + 2:]

            # update the output dependency tree
            res[rw['idx']]['link'] = lw['idx']
            res[rw['idx']]['label'] = lab
            i -= 2

        # update the number of words left
        sentlen = len(s)

        logging.debug("ACT: {:10s}".format(act))

        # increment the sentence position by 1
        i += 1

    # if we are parsing, print the final tree
    if not train:
        print_conll(res)


def main():
    # feature weights for linear classification
    # of parser actions
    wdep = defaultdict(make_defaultdict)

    train_data = []

    random.seed(123456789)

    parser = argparse.ArgumentParser()
    parser.add_argument('model_path')
    parser.add_argument('input_path')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-i', '--num_iter', type=int, default=20)
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()


    # set up logging
    log_level = logging.INFO
    if args.verbose:
        log_level = logging.DEBUG
    logging.basicConfig(format=('%(message)s'), level=log_level)

    # if we are not training, load a model
    if not args.train:
        with open(args.model_path) as f:
            wdep = json.load(f)


    # read in the data and either parse it or store it for training
    with open(args.input_path) as fp:
        s = read_sentence_conll(fp)
        sentcnt = 0
        while s:
            parse(s, wdep, train=args.train, train_data=train_data)
            sentcnt += 1
            if sentcnt % 1000 == 0:
                logging.info('{}...'.format(sentcnt))

            s = read_sentence_conll(fp)

    # if we are training, we already filled train_data with
    # everything we need to estimate the feature weights
    if args.train:
        n_train_data = len(train_data)
        wa = defaultdict(make_defaultdict)
        for t in range(0, args.num_iter):
            logging.info('Iteration {}'.format(t + 1))
            random.shuffle(train_data)
            for i in range(0, len(train_data)):
                if (i + 1) % 10000 == 0:
                    logging.info("{} / {}...".format(i + 1, n_train_data))

                prediction_action = classify(wdep, train_data[i]['feats'])

                # perceptron update
                if prediction_action != train_data[i]['label']:
                    for feat in train_data[i]['feats']:
                        wdep[train_data[i]['label']][feat] += 1.0
                        wdep[prediction_action][feat] -= 1.0

            # weight averaging (keep a sum of weights at each iteration)
            for a in wdep:
                for feat in wdep[a]:
                    wa[a][feat] = wa[a][feat] + wdep[a][feat]

        # note: weight averaging (divide by the number of iterations)
        # is not necessary since it does not affect the argmax

        with open(args.model_path, 'w') as outfile:
            json.dump(wa, outfile)


if __name__ == '__main__':
    main()
