import sys
import utility
import numpy as np
import pandas as pd

rtext_rev = 4
rtext_sgd_rev = 1
usrdelcats = ['type', 'name', 'votes_cool', 'votes_funny', 'votes_useful']
busdelcats = ['type', 'neighborhoods', 'full_address', 'latitude', 'longitude',
              'name', 'open', 'state', 'city', 'categories']
normedcols = ['business_review_count', 'review_stars', 'user_average_stars',
              'business_average_stars', 'user_review_count', 'review_length']
delcols = ['votes_useful', 'business_id', 'user_id', 'date']

def norm_col(revlist, c) :
    cmin = min(np.min(revlist[0][c]), np.min(revlist[1][c]))
    cmax = max(np.max(revlist[0][c]), np.max(revlist[1][c]))
    revlist[0][c] = (revlist[0][c] - cmin) / float(cmax - cmin)
    revlist[1][c] = (revlist[1][c] - cmin) / float(cmax - cmin)

def main():
    usr = pd.concat((utility.load_data('training', 'user'),
                     utility.load_data('test', 'user')))
    for cat in usrdelcats :
        if hasattr(usr, cat) : del usr[cat]
    usr = usr.rename(columns={'average_stars' : 'user_average_stars',
                              'review_count' : 'user_review_count'})

    bus = pd.concat((utility.load_data('training', 'business'),
                     utility.load_data('test', 'business')))
    for cat in busdelcats :
        if hasattr(bus, cat) : del bus[cat]
    bus = bus.rename(columns={'stars' : 'business_average_stars',
                              'review_count' : 'business_review_count',})

    rtxttag = 'rtext_rev{}'.format(rtext_rev)
    rtxt_tr = utility.load_data('training', rtxttag)
    rtxt_te = utility.load_data('test', rtxttag)
    rtxt_te.index = rtxt_te.index + len(rtxt_tr)
    rtxt = pd.concat((rtxt_tr, rtxt_te))

    sgdtag = 'rtext_sgd_rev{}'.format(rtext_sgd_rev)
    sgdtxt_tr = utility.load_data('training', sgdtag)
    sgdtxt_te = utility.load_data('test', sgdtag)
    sgdtxt_te.index = sgdtxt_te.index + len(sgdtxt_tr)
    sgdtxt = pd.concat((sgdtxt_tr, sgdtxt_te))

    tesrev = utility.load_data('test', 'review')
    trarev = utility.load_data('training', 'review')
    revlist = [trarev, tesrev]
    for i in range(len(revlist)) :
        revlength = revlist[i]['text'].apply(lambda t : len(t.split()))
        revlist[i]['review_length'] = revlength

        for col in revlist[i].columns :
            if not col in ['review_id', 'stars', 'date', 'review_length'] :
                del revlist[i][col]
        revlist[i] = revlist[i].rename(columns={'stars' : 'review_stars'})

        revlist[i] = pd.merge(revlist[i], rtxt, 'left')
        revlist[i] = pd.merge(revlist[i], sgdtxt, 'left')
        revlist[i] = pd.merge(revlist[i], usr, 'left')
        revlist[i] = pd.merge(revlist[i], bus, 'left')
        revlist[i] = revlist[i].fillna(-1)

    for c in normedcols : norm_col(revlist, c)

    dates = [pd.to_datetime('2013-01-19'), pd.to_datetime('2013-03-12')]
    for i in range(len(revlist)) :
        ddiff = dates[i] - revlist[i]['date']
        ddiff = ddiff.apply(lambda x: x / np.timedelta64(1, 'D'))
        revlist[i]['datediff'] = ddiff
    norm_col(revlist, 'datediff')

    for i in range(len(revlist)) :
        for c in delcols :
            if hasattr(revlist[i], c) :
                del revlist[i][c]

    utility.save_data(revlist[0], 'training', 'finalinput')
    utility.save_data(revlist[1], 'test', 'finalinput')
    
if __name__ == "__main__" :
    main()
