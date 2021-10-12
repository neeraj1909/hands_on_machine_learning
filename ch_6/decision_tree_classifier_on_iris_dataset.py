import os
import numpy as np
from pathlib import Path

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz


iris = load_iris()
X = iris.data[:, 2:]    # pedal length and width
y = iris.target
# print(iris.keys())
# print(iris['data'])
tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X, y)

# Visualization of trained Decision Tree
f = open("iris_tree.dot", 'w')
current_directory = Path(__file__).parent.absolute()
PROJECT_ROOT_DIR = "."
CHAPTER_ID = 'ch_6'

def image_path(fig_id):
    return os.path.join(current_directory, fig_id)

export_graphviz(
    tree_clf,
    out_file=image_path("iris_tree.dot"),
    feature_names=iris.feature_names[2:],
    class_names=iris.target_names,
    rounded=True,
    filled=True
)

# estimating class and class probabilities
print(tree_clf.predict_proba([[5, 1.5]]))
print(tree_clf.predict([[5, 1.5]]))
