from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import classification_report, confusion_matrix
import graphviz 

#load wine dataset and devide it on train and test samples
X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

#define classifier with max_depth=5 and min_samples_leaf=5
dt = tree.DecisionTreeClassifier(max_depth=5, min_samples_leaf=5)

#train classifier
dt.fit(X_train, y_train)

#export desision tree as pdf
dot_data = tree.export_graphviz(dt, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render('wine')

#predict wine class and calculate metrics
y_pred = dt.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))