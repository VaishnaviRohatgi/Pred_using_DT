import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import tree
import matplotlib.pyplot as plt

# Loading the dataset
df = pd.read_csv(r'C:\Users\Dell\Desktop\datasets\Iris.csv')
print(df.columns)

# Preparing feature and label
df2 = df.drop(['Id'], axis=1)
y = df2["Species"]
X = df2.drop('Species', axis=1)
print(X)

# Splitting dataset into training and testing datasets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# Creating the classifier
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train, y_train)

# Predicting the accuracy
y_pred = clf.predict(X_test)
print('score:', accuracy_score(y_pred, y_test))


print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Visualizing the decision tree
feature_names=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
target_names=['setosa', 'versicolor', 'virginica']
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=300)
tree.plot_tree(clf, feature_names=feature_names, class_names=target_names, filled=True)
fig.savefig('image_name.png')
