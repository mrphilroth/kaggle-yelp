import os
import csv
import json
import pickle
import random
import numpy as np
import pandas as pd
from glob import glob
from os.path import dirname, abspath, exists, realpath

bdir = dirname(realpath(__file__))
ddir = abspath(bdir + "/../data/")
subdir = abspath(bdir + "/../submissions/")
if not exists(ddir) : os.mkdir(ddir)
if not exists(subdir) : os.mkdir(subdir)

def identity(x): return x
converters = { "text" : identity, "name" : identity, "categories" : identity }

def rmsle(pred, actual) :
    if len(pred) != len(actual) : return None
    return np.sqrt(np.sum(np.square(np.log(pred + 1) -
                                    np.log(actual + 1))) / len(pred))

def rmsle_log(pred, actual) :
    if len(pred) != len(actual) : return None
    return np.sqrt(np.sum(np.square(pred - actual) / len(pred)))

def convert(x) :
    ''' Convert a json string to a flat python dictionary
    which can be passed into Pandas. '''
    ob = json.loads(x)
    for k, v in ob.items():
        if isinstance(v, list):
            ob[k] = ','.join(v)
        elif isinstance(v, dict):
            for kk, vv in v.items():
                ob['%s_%s' % (k, kk)] = vv
            del ob[k]
    return ob

def json_to_csv(jsonfn, csvfn) :
    print "Converting {} to {}".format(jsonfn, csvfn)
    df = pd.DataFrame([convert(line) for line in file(jsonfn)])
    df.to_csv(csvfn, encoding='utf-8', index=False)

def convert_all() :
    for jsonfn in glob("{}/*.json".format(ddir)) :
        csvfn = "{}.csv".format(jsonfn[:-5])
        json_to_csv(jsonfn, csvfn)

def load_data(tset="training", kset="review", nrand=None) :
    if not tset in ["training", "test"] : return None
    csvfn = "{}/yelp_{}_set_{}.csv".format(ddir, tset, kset)
    if not exists(csvfn) :
        jsonfn = "{}/yelp_{}_set_{}.json".format(ddir, tset, kset)
        if not exists(jsonfn) :
            print "No data found."
            return None
        else :
            print "First converting json data to csv."
            json_to_csv(jsonfn, csvfn)
    df = pd.io.parsers.read_csv(csvfn, converters=converters)
    if nrand :
        inds = random.sample(range(len(df)), nrand)
        df = df.ix[inds]
    if hasattr(df, 'date') :
        df['date'] = pd.to_datetime(df.date)
    return df

def save_data(df, tset="training", kset="output") :
    if not tset in ["training", "test"] : return None
    csvfn = "{}/yelp_{}_set_{}.csv".format(ddir, tset, kset)
    df.to_csv(csvfn, index=False)

def save_model(model, mname="random_forest_rev1"):
    modelfn = "{}/{}.pickle".format(ddir, mname)
    pickle.dump(model, open(modelfn, "w"))

def load_model(mname="random_forest_rev1"):
    modelfn = "{}/{}.pickle".format(ddir, mname)
    return pickle.load(open(modelfn))

def write_submission(dfpred, fn):
    subfn = "{}/{}".format(subdir, fn)
    for c in dfpred.columns :
        if not c in ['review_id', 'votes'] :
            del dfpred[c]
    if len(dfpred.columns) != 2 :
        print "Problem with prediction data frame"
        return None
    dfpred = dfpred.rename(columns={'review_id' : 'id'})
    dfpred.to_csv(subfn, index=False)
