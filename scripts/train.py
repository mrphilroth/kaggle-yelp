import random
import utility
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer

revision = 3
def get_pipeline():
    steps = [("classify", RandomForestRegressor(compute_importances=True,
                                                n_estimators=120,
                                                verbose=2,
                                                n_jobs=4))]
             # ("regress", SGDRegressor(loss='squared_loss',
             #                          n_iter=50))]
    return Pipeline(steps)

def main():
    print("Reading in the training data")
    train = utility.load_data("training", "finalinput")
    truth = np.ravel(np.array(train['votes_useful_log']))
    del train['votes_useful_log']

    print("Extracting features and training review text model")
    classifier = get_pipeline()
    classifier.fit(train.values[:,1:], np.array(truth))

    print("Saving the classifier")
    utility.save_model(classifier, "fullsgd_model_rev{}".format(revision))

if __name__=="__main__":
    main()
