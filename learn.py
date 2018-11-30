import load_data
import numpy as np

""" The cuisine is a frequency table which holds the real counts of ingredients in each
cuisine type based on recipes that will be used in Naive Bayes analysis
            indian  american    greek
turmeric      3         1         0
tomato        2         2         1
"""
cuisine_map = {}
cuisine_total_counts = {}

def encodeRecipes(jsonFile):
    dataset = load_data.loadDataSet(jsonFile)

    recipesList = dataset['ingredients']
    ingredientsList = list(dataset['ingredientsSet'])

    """ 2-D array storing a representation of each recipe. Each recipe is
    encoded as an array of 1's and 0's representing the ingredients it does and
    doesn't have. The order of the ingredients matches the order of
    ingredientsList above."""
    encodedRecipes = []

    for idx, r in enumerate(recipesList):
        encodedRecipe = []
        for i in ingredientsList:
            encodedRecipe.append('1' if i in r else '0')
        encodedRecipes.append(encodedRecipe)
        print "Finished %i of 39,773." % idx

    return encodedRecipes


def writeRecipes(encodedRecipes):
    np.savetxt("recipe_arrays.txt", encodedRecipes, fmt="%s")

def naiveBayes():
    trainData("train.json")
    test_data = encodeRecipes("test.json")


def trainData(jsonFile):
    recipeData = loadJson(jsonFile)

    for recipe in recipeData:
        cuisine_type = recipe["cuisine"]
        for i in recipe["ingredients"]:
            if not i in cuisine_map:
                cuisine_map[i] = {}
                cuisine_map[i]["total"] = 0

            if not cuisine_type in cuisine_map[i]:
                cuisine_map[i][cuisine_type] = 0

            if not cuisine_type in cuisine_total_counts:
                cuisine_total_counts[cuisine_type] = 0

            cuisine_map[i][cuisine_type] += 1
            cuisint_map[i]["total"] +=1
            cuisine_total_counts[cuisine_type] +=1

def loadRecipes():
    encodedRecipes = np.loadtxt("recipe_arrays.txt")
    return encodedRecipes
