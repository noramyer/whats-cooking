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


def loadDataSet(jsonFile):
    jsonObj = loadJson(jsonFile)
    dataSet = {'ids': loadIds(jsonObj), 'cuisines': loadCuisines(jsonObj),
               'ingredients': loadIngredients(jsonObj)}

    print(dataSet)


def main():
    loadDataSet('train.json')


if __name__ == "__main__":
    main()
