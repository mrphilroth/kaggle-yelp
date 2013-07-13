import random
import utility
from sklearn.pipeline import Pipeline
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer

def get_pipeline():
    steps = [("vectorize", HashingVectorizer(n_features=2**16,
                                             stop_words='english',
                                             charset_error='ignore')),
             ("cluster", MiniBatchKMeans(n_clusters=500,
                                         batch_size=400,
                                         verbose=1)),
             ("classify", RandomForestRegressor(n_estimators=100,
                                                verbose=2,
                                                n_jobs=4))]
    return Pipeline(steps)

def main():
    revision = 4

    print("Reading in the training data")
    train = utility.load_data("training", "rtext")
    inds = random.sample(range(len(train)), 100000)
    mtrain = train.ix[inds]

    print("Extracting features and training review text model")
    classifier = get_pipeline()
    classifier.fit(list(mtrain['rtext_bcat']), 
                   list(mtrain['votes_useful_log']))

    print("Saving the classifier")
    utility.save_model(classifier, "train_rtext_rev{}".format(revision))

if __name__=="__main__":
    main()
