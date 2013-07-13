import random
import utility
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import HashingVectorizer

def get_pipeline():
    steps = [("vectorize", HashingVectorizer(n_features=2**18,
                                             stop_words='english',
                                             charset_error='ignore')),
             ("classify", SGDClassifier(verbose=2))]
    return Pipeline(steps)

def main():
    revision = 1

    print("Reading in the training data")
    train = utility.load_data("training", "rtext")

    print("Extracting features and training review text model")
    classifier = get_pipeline()
    classifier.fit(list(train['rtext_bcat']), list(train['votes_useful_log']))

    print("Saving the classifier")
    utility.save_model(classifier, "train_rtext_sgd_rev{}".format(revision))

if __name__=="__main__":
    main()
