import utility
import numpy as np
import pandas as pd

def main():
    revision = 1

    print("Loading the classifier")
    classifier = utility.load_model("train_rtext_rev{}".format(revision))
    
    print("Reading in the training data")
    train = utility.load_data("training", "rtext")

    print("Predicting the rest of the training data")
    pred = np.ravel(classifier.predict(list(train['rtext_bcat'])))
    score = utility.rmsle_log(pred, train['votes_useful_log'])
    print "Score:", score

    print("Writing out new training data")
    del train['rtext_bcat']
    train['votes_useful_log_rtextpred_sgd'] = pd.Series(pred, index=train.index)
    utility.save_data(train, "training", "rtext_sgd_rev{}".format(revision))
    
    print("Reading in the test data")
    test = utility.load_data("test", "rtext")
    tepred = np.ravel(classifier.predict(list(test['rtext_bcat'])))

    print("Writing out new test data")
    del test['rtext_bcat']
    test['votes_useful_log_rtextpred_sgd'] = pd.Series(tepred, index=test.index)
    utility.save_data(test, "test", "rtext_sgd_rev{}".format(revision))
    test['votes'] = pd.Series(np.exp(tepred) + 1, index=test.index)

    print("Writing out a new submission file")
    utility.write_submission(test, "rtextsgd_sub_rev{}".format(revision))

if __name__=="__main__":
    main()
