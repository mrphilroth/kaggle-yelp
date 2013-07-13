import sys
import string
import utility
import numpy as np
import pandas as pd

delrevcats = ["date", "stars", "type", "votes_cool", "votes_funny"]
delbuscats = ["city", "full_address", "latitude", "longitude", "name",
              "neighborhoods", "open", "review_count", "stars", "state", "type"]

def process_bcat(bcatstr) :
    toret = ""
    for bcat in  bcatstr.split(',') :
        for w in bcat.split(' ') :
            w = w.strip('()& ')
            if w : toret += " bcat{}".format(w)
    return toret

def main():

    trabus = utility.load_data("training", "business")
    tesbus = utility.load_data("test", "business")
    bus = pd.concat((trabus, tesbus))
    for cat in delbuscats :
        if hasattr(bus, cat) : del bus[cat]
    bus['procbcat'] = pd.Series(map(process_bcat, bus['categories']), bus.index)
    del bus['categories']

    for s in ["training", "test"] :

        rev = utility.load_data(s, "review")
        for cat in delrevcats :
            if hasattr(rev, cat) : del rev[cat]
        if hasattr(rev, 'votes_useful') :
            rev['votes_useful_log'] = np.log(rev.votes_useful + 1)
        rev = pd.merge(rev, bus, 'inner')

        rev['rtext_bcat'] = rev['text'] + rev['procbcat']
        del rev['procbcat']
        del rev['text']

        utility.save_data(rev, s, 'rtext')
    
if __name__ == "__main__" :
    main()
