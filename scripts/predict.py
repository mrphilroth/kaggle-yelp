import utility
import numpy as np

revision = 3
def main():
    print("Loading the classifier")
    classifier = utility.load_model("fullsgd_model_rev{}".format(revision))

    print("Reading in the training data")
    train = utility.load_data("training", "finalinput")
    truth = train['votes_useful_log']
    del train['votes_useful_log']

    print("Predicting the training data")
    logpred = np.ravel(classifier.predict(train.values[:,1:]))
    score = utility.rmsle_log(logpred, truth)
    print "Score:", score

    print("Reading in the test data")
    test = utility.load_data("test", "finalinput")
    del test['votes_useful_log']

    print("Predicting the test data")
    logpred = np.ravel(classifier.predict(test.values[:,1:]))
    pred = np.exp(np.array(logpred, dtype=np.float64)) - 1
    test['votes'] = pred
    
    print("Writing out a new submission file")
    utility.write_submission(test, "fullsgd_sub_rev{}.csv".format(revision))

if __name__=="__main__":
    main()
