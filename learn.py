import load_data
import numpy as np
from sklearn.naive_bayes import MultinomialNB, metrics

""" The cuisine_freq_table is a frequency table which holds the real counts of ingredients in each
cuisine type based on recipes that will be used in Naive Bayes analysis
            indian  american    greek
turmeric      3         1         0
tomato        2         2         1
"""
cuisine_freq_table = {}

""" The cuisine_likelihood_table is a likelihood table which holds the real counts of ingredients in each
cuisine type based on recipes that will be used in Naive Bayes analysis
            indian  american    greek
turmeric      3/5       1/3       0/1
tomato        2/5       2/3       1/1
"""
cuisine_likelihood_table = {}
cuisine_total_counts = {}

def encodeRecipes(jsonFile):
    dataset = load_data.loadDataSet(jsonFile)
    jsonObj = loadJson(jsonFile)

    recipesList = dataset['ingredients']
    ingredientsList = list(dataset['ingredientsSet'])
    cuisineList = dataset['cuisinesSet']
    cuisineMap = {}

    for j in len(cuisineList):
        cuisineMap[j] = cuisineList[j]

    """ 2-D array storing a representation of each recipe. Each recipe is
    encoded as an array of 1's and 0's representing the ingredients it does and
    doesn't have. The order of the ingredients matches the order of
    ingredientsList above."""
    encodedRecipes = []
    encodedResults = []

    for idx, dish in enumerate(jsonObj):
        encodedRecipe = []
        encodedResults[idx] = cuisineMap[dish["cuisine"]]
        for i in ingredientsList:
            encodedRecipe.append('1' if i in dish["ingredients"] else '0')
        encodedRecipes.append(encodedRecipe)
        print "Finished %i of 39,773." % idx

    return encodedRecipes, encodedResults, cuisineMap, len(jsonObj)


def writeRecipes(encodedRecipes):
    np.savetxt("recipe_arrays.txt", encodedRecipes, fmt="%s")

def naiveBayes():
    nb_model = MultinomialNB()
    x_train, y_train = encodeRecipes("train.json")
    x_test, y_test, cuisine_map, recipe_total_count = encodeRecipes("test.json")

    accuracy = train_model(nb_model, x_train, y_train, x_test, y_test)
    print("Recipe classification complete. Getting accuracy results now ...")
    print("MultinomialNB accuracy for %i recipes = %f" %recipe_total_count, accuracy)

def loadRecipes():
    encodedRecipes = np.loadtxt("recipe_arrays.txt")
    return encodedRecipes

def train_model(classifier, x_train, y_train, test_matrix, test_matrix_expected_results):
    classifier.fit(x_train, y_train)

    predictions = classifier.predict(test_matrix)

    return metrics.accuracy_score(predictions, test_matrix_expected_results)
