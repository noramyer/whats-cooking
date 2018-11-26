import load_data
import numpy as np


def encodeRecipes():
    dataset = load_data.loadDataSet('train.json')

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


def loadRecipes():
    encodedRecipes = np.loadtxt("recipe_arrays.txt")
    return encodedRecipes
