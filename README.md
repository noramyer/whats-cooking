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

```
