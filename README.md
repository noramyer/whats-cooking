# whats-cooking
Machine learning on recipe ingredients

### Using the datasets
4 different data sets:
  - tiny_train.json #small, 3-cuisine typed dataset to quickly fit models
  - tiny_test.json #small, 3-cuisine types recipes to quickly predict with classifiers
  - train.json #the main data set,around 26,000 classified recipes
  - test.json #main test data for classifiers, about 14,000 recipes

Use train.json and test.json for running the classifiers

### About the data
Total Number of Recipes:  25580

Number of Cuisines:  20

Highest Occurring Cuisines:
  ('italian', 5024)
  ('mexican', 4130)
  ('southern_us', 2757)

Number of Ingredients:  5955

Highest Occurring Ingredients:
  ('salt', 11629)
  ('onions', 5162)
  ('olive oil', 5145)

Check out *data_analysis.txt* for a full analysis of the data set including means, medians, averages, and the top 10 ingredients for every cuisine.

Data format:
```
{
  "id": 10259,
  "cuisine": "greek",
  "ingredients": [
    "romaine lettuce",
    "black olives",
    "grape tomatoes",
    "garlic",
    "pepper",
    "purple onion",
    "seasoning",
    "garbanzo beans",
    "feta cheese crumbles"
  ]
}
```

### Data representations
There are two different data representations used with naive bayes, SGD, and logistic regression models.
Once ran, wait for the data to encode
```
Encoding training data. Progress:
[=====================================-----------------------] 61.2%
```

#### Feature Vector
To see classifier accuracies with feature vector representation, run:
```
$python feature_vector_learn.py
```

#### TF-IDF
To see classifier accuracies with tfidf_learn, run:
```
$python tfidf_learn.py
```
### Loading data set

```
import load_data

...

dataset = load_data.loadDataSet('train.json')

ids = dataset['ids'] # numpy array of each id
cuisines = dataset['cuisines'] # numpy array of each cuisine
ingredients = dataset['ingredients'] # 2D Numpy Array of ingredients

cuisinesSet = dataset['cuisinesSet'] # set of the 20 cuisines represented
cuisinesCount = dataset['cuisinesCount'] # frequency of each of the 20 cuisines in the dataset
ingredientsSet = dataset['ingredientsCount'] # set of the 6714 ingredients represented
ingredientsCount = dataset['ingredientsCount'] # frequency of each of the 6714 ingredients in the dataset
```

### Creating arrays of ingredients for recipes

```
import learn

recipes = learn.encodeRecipes()
```
