kaggle-yelp
===========

The code I used to participate in Kaggle's Yelp Recruiting
contest. Find my profile
[here](http://www.kaggle.com/users/25160/proth).

Yelp wanted to be able to predict what reviews would be deemed useful
by their user base before those reviews were made public. They
packaged a bunch of their data up for this competition. I spent a lot
of time learning about different strategies for sparse text
vectorization and what scikit-learn algorithms could run on that
sparse input. My final model focused on optimally using the vectorized
text which the winners then revealed didn't contain much useful
information.

I added unique identifiers to each review text in order to add
information about the business category. I then used a hashing
vectorizer on the review text. I applied two models to the sparse,
vectorized text: a random forest regressor and a stochastic gradient
descent (SGD) regressor. The SGD regressor was able to handle the
sparse input.  The random forest regressor was not, and so I found as
many cluster centers as would fit in my laptop's RAM and calculated
each review's distance to those cluster centers. That was then the
data that I fed to the random forest regressor.

I used the output of each of those regressors in a new random forest
model along with user and business information (number of previous
user reviews, average business rating, etc...). That was then the
final predictor. It turned out that people had the most success
creating features from the user and business information and only
including general description stats about the review text (length,
number of paragraphs, etc...).
