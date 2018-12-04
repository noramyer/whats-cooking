# from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
import numpy as np
from pprint import pprint

from load_data import loadDataSet


# def tutorialLearn():
#     categories = ['alt.atheism', 'soc.religion.christian',
#                   'comp.graphics', 'sci.med']

#     twenty_train = fetch_20newsgroups(subset='train',
#                                       categories=categories, shuffle=True, random_state=42)

#     # Extracting features from text files
#     # -- Tokenizing text with CountVectorizer
#     # -- index value of word in vocabulary linked to frequency in WHOLE training corpus

#     count_vect = CountVectorizer()
#     X_train_counts = count_vect.fit_transform(twenty_train.data)

#     # Transform occurrences to frequences with TF-IDF (term-frequency and inverse document frequency)
#     # what the heck is that? https://en.wikipedia.org/wiki/Tf%E2%80%93idf

#     tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
#     X_train_tf = tf_transformer.transform(X_train_counts)
#     X_train_tf = tf_transformer.transform(X_train_counts)

#     # Training classifier
#     tfidf_transformer = TfidfTransformer()
#     X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

#     # adding naive bayes to the whole thing
#     clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)

#     # some tests
#     docs_new = ['God is love', 'OpenGL on the GPU is fast']
#     X_new_counts = count_vect.transform(docs_new)
#     X_new_tfidf = tfidf_transformer.transform(X_new_counts)

#     # results
#     predicted = clf.predict(X_new_tfidf)

#     for doc, category in zip(docs_new, predicted):
#         print('%r => %s' % (doc, twenty_train.target_names[category]))

#     # Naive Bayes Classifier
#     text_clf = Pipeline([
#         ('vect', CountVectorizer()),
#         ('tfidf', TfidfTransformer()),
#         ('clf', MultinomialNB())
#     ])

#     text_clf.fit(twenty_train.data, twenty_train.target)

#     twenty_test = fetch_20newsgroups(
#         subset='test', categories=categories, shuffle=True, random_state=42)

#     docs_test = twenty_test.data
#     predicted = text_clf.predict(docs_test)
#     accuracy = np.mean(predicted == twenty_test.target)
#     print('Naive Bayes Accuracy: %s' % str(accuracy))

#     # Stochastic Gradient Descent Algorithm
#     text_clf = Pipeline([
#         ('vect', CountVectorizer()),
#         ('tfidf', TfidfTransformer()),
#         ('clf', SGDClassifier(loss='hinge', penalty='l2',
#                               alpha=1e-3, random_state=42,
#                               max_iter=5, tol=None))
#     ])

#     text_clf.fit(twenty_train.data, twenty_train.target)
#     predicted = text_clf.predict(docs_test)
#     accuracy = np.mean(predicted == twenty_test.target)
#     print('SGD Accuracy: %s' % str(accuracy))

# assigns a numerical identifier to each label


def processLabels(labels):
    formattedLabels = []
    uniqueLabels = []

    for label in labels:
        if label not in uniqueLabels:
            uniqueLabels.append(label)
        formattedLabels.append(uniqueLabels.index(label))
    return formattedLabels


def preProcessData(data):
    return [' '.join(str(ingredient)
                     for ingredient in [item.replace(' ', '-') for item in row]) for row in data]


def learnCuisines(trainingData, trainingLabels, testData, testLabels, SGD=False):
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(trainingData)

    tfidf_transformer = TfidfTransformer()

    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    clf = MultinomialNB().fit(X_train_tfidf, trainingLabels)

    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB())
    ])

    text_clf.fit(trainingData, trainingLabels)

    predicted = text_clf.predict(testData)

    result = np.mean(predicted == testLabels)
    print('naive bayes: %s' % str(result))

    if SGD:
        print('hinge loss:')
        text_clf = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                  alpha=1e-6, random_state=42,
                                  max_iter=200, tol=None))
        ])

        text_clf.fit(trainingData, trainingLabels)

        predicted = text_clf.predict(testData)

        result = np.mean(predicted == testLabels)
        print('hinge loss accuracy: %s' % str(result))

        text_clf = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', SGDClassifier(loss='log', penalty='l2',
                                  alpha=1e-6, random_state=42,
                                  max_iter=100, tol=None))
        ])

        text_clf.fit(trainingData, trainingLabels)

        predicted = text_clf.predict(testData)

        result = np.mean(predicted == testLabels)
        print('logistic loss accuracy: %s' % str(result))


def main():
    trainingData = loadDataSet('train.json')
    ingredients = trainingData['ingredients']
    cuisines = trainingData['cuisines']
    formattedIngredients = preProcessData(ingredients)
    formattedLabels = processLabels(cuisines)

    trainingIngredients = formattedIngredients[0: len(formattedIngredients)/2]
    trainingLabels = formattedLabels[0: len(formattedIngredients)/2]

    testIngredients = formattedIngredients[len(formattedIngredients)/2:]
    testLabels = formattedLabels[len(formattedIngredients)/2:]

    learnCuisines(trainingIngredients, trainingLabels,
                  testIngredients, testLabels, True)


if __name__ == '__main__':
    main()
