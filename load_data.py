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
    # ingredients = np.array([stringify(dish['ingredients'])
    #                         for dish in jsonObj])

    ingredients = np.array(
        [np.array(stringify(dish['ingredients'])) for dish in jsonObj])

    return ingredients


def loadDataSet(jsonObj):
    dataSet = {'ids': loadIds(jsonObj), 'cuisines': loadCuisines(jsonObj),
               'ingredients': loadIngredients(jsonObj)}

    print(dataSet)


def main():
    jsonObj = loadJson("train.json")

    # cuisines = loadCuisines(jsonObj)
    ingredients = loadIngredients(jsonObj)
    print ingredients
    # ids = loadIds(jsonObj)


if __name__ == "__main__":
    main()
