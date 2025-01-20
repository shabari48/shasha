
# Py-Shasha

Shasha is a collection of machine learning models implemented from scratch. This library provides simple and easy-to-use implementations of various machine learning algorithms, including linear regression, multiple regression, logistic regression, k-nearest neighbors (KNN), decision trees,random forests,XGB classifier and SVM.

## Installation

You can install Shasha using pip:

```bash
pip install shasha
```

## Usage

Here are some examples of how to use the models provided by Shasha:

### Linear Regression

```python
from shasha import My_Linear_Regression
import numpy as np

# Sample data
X_train = np.array([[1], [2], [3]])
y_train = np.array([1, 2, 3])

# Create and train the model
model = My_Linear_Regression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_train)
print(predictions)
```

### Multiple Regression

```python
from shasha import My_Multiple_Regression
import numpy as np

# Sample data
X_train = np.array([[1, 2], [2, 3], [3, 4]])
y_train = np.array([1, 2, 3])

# Create and train the model
model = My_Multiple_Regression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_train)
print(predictions)
```

### Logistic Regression

```python
from shasha import My_Logistic_Regression
import numpy as np

# Sample data
X_train = np.array([[1, 2], [2, 3], [3, 4]])
y_train = np.array([0, 1, 0])

# Create and train the model
model = My_Logistic_Regression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_train)
print(predictions)
```

### K-Nearest Neighbors (KNN)

```python
from shasha import My_KNN
import numpy as np

# Sample data
X_train = np.array([[1, 2], [2, 3], [3, 4]])
y_train = np.array([0, 1, 0])
X_test = np.array([[2, 2]])

# Create and train the model
model = My_KNN(n_neighbors=3)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
print(predictions)
```

### Random Forest

```python
from shasha import My_Random_Forest
import numpy as np

# Sample data
X_train = np.array([[1, 2], [2, 3], [3, 4]])
y_train = np.array([0, 1, 0])
X_test = np.array([[2, 2]])

# Create and train the model
model = My_Random_Forest(n_trees=10)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
print(predictions)
```

### Decision Tree

```python
from shasha import My_DecisionTree
import numpy as np

# Sample data
X_train = np.array([[1, 2], [2, 3], [3, 4]])
y_train = np.array([0, 1, 0])
X_test = np.array([[2, 2]])

# Create and train the model
model = My_DecisionTree()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
print(predictions)

```

### XGBoost Classifier

```python
from py-shasha import My_XGB_Classifier
import numpy as np

# Sample data
X_train = np.array([[1, 2], [2, 3], [3, 4]])
y_train = np.array([0, 1, 0])
X_test = np.array([[2, 2]])

# Create and train the model
model = My_XGB_Classifier()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
print(predictions)
```

### Support Vector Machine (SVM)

```python
from shasha import My_SVM
import numpy as np

# Sample data
X_train = np.array([[1, 2], [2, 3], [3, 4]])
y_train = np.array([0, 1, 0])
X_test = np.array([[2, 2]])

# Create and train the model
model = My_SVM()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
print(predictions)
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on GitHub.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact

For any questions or feedback, please contact [Shabari Prakash](mailto:shabariprakashsv@gmail.com).
