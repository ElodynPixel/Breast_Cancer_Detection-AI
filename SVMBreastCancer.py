# Importing sklearn to use SVM & a breast cancer dataset.
import sklearn
from sklearn import svm
from sklearn import datasets

# Breast cancer dataset.
BCancer = datasets.load_breast_cancer()

# List of the features and labels in the dataset.
print(BCancer.feature_names)
print(BCancer.target_names)


xFeatures=BCancer.data # All of the features
yLabels=BCancer.target # All of the labels

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(xFeatures, yLabels, test_size=0.5)

print(x_train, y_train)

# Remove # below if you want to see data.
# print(x_train[:5], y_train[:5])

classes=('malignant' 'benign')
