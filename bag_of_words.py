from load_data import loadDataSet
from pprint import pprint
import operator

training_data = loadDataSet('train.json')

print(training_data['cuisines'])

# for item in training_data:
#     print(item)


# just a good snippet to know. . .
sorted_by_value = sorted(
    training_data['ingredientsCount'].items(), key=lambda kv: kv[1])

# training_data
# -- cuisinesSet
# -- ingredients
# -- cuisines
# -- cuisinesCount
# -- ids
# -- ingredientsCount
# -- ingredientsSet

# print(len(training_data['cuisines']))
# print((training_data['ingredients']))

cuisineBag = {cuisine: set() for cuisine in training_data['cuisinesSet']}


# this constructs a dictionary of the form
# {
#    cuisine: set([ ingredient, .... , ingredient ]),
#      ...
#       cuisine: set([ ingredient, .... , ingredient ]),
# }
for dishIndex in range(len(training_data['cuisines'])):
    cuisine = training_data['cuisines'][dishIndex]
    cuisineBag[cuisine].update(
        set([str(item).upper() for item in training_data['ingredients'][dishIndex]]))

print(cuisineBag)
for cuisine in cuisineBag:
    cuisineBag[cuisine] = list(cuisineBag[cuisine])
