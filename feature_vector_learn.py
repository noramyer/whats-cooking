import load_data
import numpy as np
import json
from sklearn import linear_model, metrics, naive_bayes, svm
import sys
import warnings

warnings.filterwarnings('ignore', category=UnicodeWarning)

cuisineMap = {}
ingredientsList = []
xTrain = []
yTrain = []
xTest = []
yTest = []
testRecipeTotalCount = 0

def printProgress(count, total):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    if count != total:
        sys.stdout.write('[%s] %s%s\r' % (bar, percents, '%'))
    else:
        sys.stdout.write('[%s] %s%s\n' % (bar, percents, '%'))
    sys.stdout.flush()

def encodeTrainRecipes(jsonFile):
    print "Encoding training data. Progress: "
    dataset = load_data.loadDataSet(jsonFile)
    jsonObj = load_data.loadJson(jsonFile)

    recipesLength = len(jsonObj)
    ingredientsList = list(dataset['ingredientsSet'])
    global ingredientsList
    ingredientsList = ingredientsList
    cuisineList = list(dataset['cuisinesSet'])

    for j in range(len(cuisineList)):
        cuisineMap[cuisineList[j]] = j

    """ 2-D array storing a representation of each recipe. Each recipe is
    encoded as an array of 1's and 0's representing the ingredients it does and
    doesn't have. The order of the ingredients matches the order of
    ingredientsList above."""
    encodedRecipes = []
    encodedResults = []

    for idx, dish in enumerate(jsonObj):
        encodedRecipe = []
        encodedResults.append(cuisineMap[dish["cuisine"]])
        for i in ingredientsList:
            encodedRecipe.append(1 if i in dish["ingredients"] else 0)
        encodedRecipes.append(encodedRecipe)
        printProgress(idx + 1, recipesLength)

    global xTrain
    xTrain = encodedRecipes
    global yTrain
    yTrain = encodedResults

def encodeTestRecipes(jsonFile, ingredientsList, cuisineMap):
    print "Encoding test data. Progress: "
    jsonObj = load_data.loadJson(jsonFile)

    recipesLength = len(jsonObj)
    encodedRecipes = []
    encodedResults = []

    for idx, dish in enumerate(jsonObj):
        encodedRecipe = []

        if not dish["cuisine"] in cuisineMap:
            continue

        encodedResults.append(cuisineMap[dish["cuisine"]])

        for i in ingredientsList:
            encodedRecipe.append(1 if i in dish["ingredients"] else 0)
        encodedRecipes.append(encodedRecipe)
        printProgress(idx + 1, recipesLength)

    global xTest
    xTest = encodedRecipes
    global yTest
    yTest = encodedResults
    global testRecipeTotalCount
    testRecipeTotalCount = len(jsonObj)

def naiveBayes():
    print "Beginning Naive Bayes analysis."
    nbModel = naive_bayes.MultinomialNB()
    accuracy = train_model(nbModel, xTrain, yTrain, xTest, yTest)
    print "Multinomial Naive Bayes accuracy for %i recipes = %f" % (testRecipeTotalCount, accuracy)

def sgd():
    print "Beginning stochastic gradient descent analysis."
    # TODO check parameters for sgdModel
    sgdModel = linear_model.SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=100, tol=1e-3)
    accuracy = train_model(sgdModel, xTrain, yTrain, xTest, yTest)
    print "Stochastic gradient descent accuracy for %i recipes = %f" % (testRecipeTotalCount, accuracy)

def logisticRegression():
    print "Beginning logistic regression analysis."
    lrModel = linear_model.LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=200)
    accuracy = train_model(lrModel, xTrain, yTrain, xTest, yTest)
    print "Logistic regression accuracy for %i recipes = %f" % (testRecipeTotalCount, accuracy)

def train_model(classifier, x_train, y_train, x_test, y_test):
    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_test)
    return metrics.accuracy_score(predictions, y_test)

def main():
    encodeTrainRecipes("train.json")
    encodeTestRecipes("test.json", ingredientsList, cuisineMap)
    print "\nEncoding finished for recipes. Beginning analysis.\n"

    naiveBayes()
    sgd()
    logisticRegression()

if __name__ == '__main__':
    main()
