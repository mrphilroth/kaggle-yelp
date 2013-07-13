import utility
import numpy as np
import pandas as pd

def main():
    revision = 4

    print("Loading the classifier")
    classifier = utility.load_model("train_rtext_rev{}".format(revision))
    
    print("Reading in the training data")
    train = utility.load_data("training", "rtext")

    print("Predicting the rest of the training data")
    bunch = 50000
    pred = np.zeros(len(train))
    for ibunch in range(int(len(train) / bunch)) :
        beg = ibunch * bunch
        end = (ibunch + 1) * 50000
        mtrain = train.ix[beg:end - 1]
        mpred = np.ravel(classifier.predict(list(mtrain['rtext_bcat'])))
        pred[beg:end] = mpred

    beg = int(len(train) / bunch) * bunch
    mtrain = train.ix[beg:]
    mpred = np.ravel(classifier.predict(list(mtrain['rtext_bcat'])))
    pred[beg:] = mpred

    score = utility.rmsle_log(pred, train['votes_useful_log'])
    print "Score:", score

    print("Writing out new training data")
    del train['rtext_bcat']
    train['votes_useful_log_rtextpred'] = pd.Series(pred, index=train.index)
    utility.save_data(train, "training", "rtext_rev{}".format(revision))
    
    print("Reading in the test data")
    test = utility.load_data("test", "rtext")
    tepred = np.ravel(classifier.predict(list(test['rtext_bcat'])))

    print("Writing out new test data")
    del test['rtext_bcat']
    test['votes_useful_log_rtextpred'] = pd.Series(tepred, index=test.index)
    utility.save_data(test, "test", "rtext_rev{}".format(revision))
    test['votes'] = pd.Series(np.exp(tepred) + 1, index=test.index)

    print("Writing out a new submission file")
    utility.write_submission(test, "rtextrf_sub_rev{}.csv".format(revision))

if __name__=="__main__":
    main()
