import json
import numpy as np


def loadJson(jsonFile):
    with open(jsonFile) as f:
        return json.loads(f.read())


def loadIds(jsonObj):
    ids = np.array([str(dish['id']) for dish in jsonObj])
    return ids


def loadCuisines(jsonObj):
    cuisines = np.array([str(dish['cuisine']) for dish in jsonObj])
    return cuisines


def stringify(ingredients):
    return [ingredient.encode('utf-8') for ingredient in ingredients]


def loadIngredients(jsonObj):
    ingredients = np.array(
        [np.array(stringify(dish['ingredients'])) for dish in jsonObj])
    return ingredients


def countIngredients(ingredientsArray):
    ingredientsCount = {}
    for recipe in ingredientsArray:
        for i in recipe:
            if i in ingredientsCount:
                ingredientsCount[i] += 1
            else:
                ingredientsCount[i] = 1
    return ingredientsCount


def countCuisines(cuisinesArray):
    cuisinesCount = {}
    for c in cuisinesArray:
        if c in cuisinesCount:
            cuisinesCount[c] += 1
        else:
            cuisinesCount[c] = 1
    return cuisinesCount


def loadDataSet(jsonFile):
    jsonObj = loadJson(jsonFile)

    cuisines = loadCuisines(jsonObj)
    ids = loadIds(jsonObj)
    ingredients = loadIngredients(jsonObj)

    dataSet = {'ids': ids,
               'cuisines': cuisines,
               'ingredients': ingredients,
               'cuisinesSet': set(cuisines),
               'ingredientsCount': countIngredients(ingredients),
               'cuisinesCount': countCuisines(cuisines)}
    return dataSet
