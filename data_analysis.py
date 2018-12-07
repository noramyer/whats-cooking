import numpy as np

import load_data

### READ IN DATA
dataset = load_data.loadDataSet('train.json')

ids = dataset['ids'] # numpy array of each id
cuisines = dataset['cuisines'] # numpy array of each cuisine
ingredients = dataset['ingredients'] # 2D Numpy Array of ingredients

cuisinesSet = dataset['cuisinesSet'] # set of the 20 cuisines represented
cuisinesCount = dataset['cuisinesCount'] # frequency of each of the 20 cuisines in the dataset
ingredientsSet = dataset['ingredientsSet'] # set of the 6714 ingredients represented
ingredientsCount = dataset['ingredientsCount'] # frequency of each of the 6714 ingredients in the dataset

cuisineIngredientCount = {}
for cuisine, ingreds in zip(cuisines,ingredients):
    if cuisine not in cuisineIngredientCount:
        cuisineIngredientCount[cuisine] = {}
    for ingredient in ingreds:
        if ingredient not in cuisineIngredientCount[cuisine]:
            cuisineIngredientCount[cuisine][ingredient] = 1
        else:
            cuisineIngredientCount[cuisine][ingredient] += 1


print "Total Number of Recipes: ", len(ids)
print

print "Number of Cuisines: ", len(cuisinesSet)
cuisineOccurrences = cuisinesCount.values()
print "Mean number of cuisine occurrences: ", np.mean(cuisineOccurrences)
print "Median number of cuisine occurrences: ", np.median(cuisineOccurrences)
print "Standard deviation of cuisine occurrences: ", np.std(cuisineOccurrences)
sortedCuisines = sorted(cuisinesCount.items(), key=lambda x: x[1], reverse=True)
print "(Most common cuisine, Number of occurrences): ", sortedCuisines[0]
print "(Least common cuisine, Number of occurrences): ", sortedCuisines[-1]
print "All cuisines sorted by number of occurrences: "
for c in sortedCuisines:
    print c
    
print
print "Number of Ingredients: ", len(ingredientsSet)
ingredientOccurrences = ingredientsCount.values()
print "Mean number of ingredient occurrences: ", np.mean(ingredientOccurrences)
print "Median number of ingredient occurrences: ", np.median(ingredientOccurrences)
print "Standard deviation of ingredient occurrences: ", np.std(ingredientOccurrences)
sortedIngredients = sorted(ingredientsCount.items(), key=lambda x: x[1], reverse=True)
print "(Most common ingredient, Number of occurrences): ", sortedIngredients[0]
print "(Least common ingredient, Number of occurrences): ", sortedIngredients[-1]
print "Number of ingredients that appear only once: ", len([ing for ing in ingredientsCount.items() if ing[1] == 1])
print "Top 50 most common ingredients: "
for c in sortedIngredients[:50]:
    print c

print
print "TOP 10 INGREDIENTS PER CUISINE"
for cuisine in cuisineIngredientCount:
    print
    print "CUISINE: ", cuisine
    sortedIngredients = sorted(cuisineIngredientCount[cuisine].items(), key=lambda x: x[1], reverse=True)
    for c in sortedIngredients[:10]:
        print c

