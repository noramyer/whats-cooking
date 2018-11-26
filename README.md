# whats-cooking
Machine learning on recipe ingredients

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
ingredientsCount = dataset['ingredientsCount'] # frequency of each of the ~6000 ingredients in the dataset
```
