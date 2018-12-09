from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.pipeline import Pipeline
import numpy as np
from pprint import pprint
from load_data import loadDataSet

def processLabels(labels):
    formattedLabels = []
    uniqueLabels = []

    for label in labels:
        if label not in uniqueLabels:
            uniqueLabels.append(label)
        formattedLabels.append(uniqueLabels.index(label))
    return formattedLabels

def processData(data):
    return [' '.join(str(ingredient)
                     for ingredient in [item.replace(' ', '-') for item in row]) for row in data]

def naiveBayes(trainingData, trainingLabels, testData, testLabels):
    print "Beginning Naive Bayes analysis."

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

    print('Naive Bayes accuracy: %s' % str(result))

def sgd(trainingData, trainingLabels, testData, testLabels):
    print "Beginning stochastic gradient descent analysis."

    for function in ['hinge', 'log', 'perceptron']:
        text_clf = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', SGDClassifier(loss=function, penalty='l2',
                                  alpha=1e-6, random_state=42,
                                  max_iter=200, tol=1e-3))
        ])
        text_clf.fit(trainingData, trainingLabels)

        predicted = text_clf.predict(testData)
        result = np.mean(predicted == testLabels)
        print('SGD %s loss accuracy: %s' % (function, str(result)))

def logisticRegression(trainingData, trainingLabels, testData, testLabels):
    print "Beginning logistic regression analysis."

    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=200))
    ])
    text_clf.fit(trainingData, trainingLabels)

    predicted = text_clf.predict(testData)
    result = np.mean(predicted == testLabels)
    print('Logistic Regression accuracy: %s' % str(result))

def main():
    trainingData = loadDataSet('train.json')
    ingredients = processData(trainingData['ingredients'])
    labels = processLabels(trainingData['cuisines'])

    trainingIngredients = ingredients[0:len(ingredients)/2]
    trainingLabels = labels[0:len(ingredients)/2]
    testIngredients = ingredients[len(ingredients)/2:]
    testLabels = labels[len(ingredients)/2:]
    print "Encoding finished for recipes. Beginning analysis.\n"

    naiveBayes(trainingIngredients, trainingLabels, testIngredients, testLabels)
    sgd(trainingIngredients, trainingLabels, testIngredients, testLabels)
    logisticRegression(trainingIngredients, trainingLabels, testIngredients, testLabels)

if __name__ == '__main__':
    main()
