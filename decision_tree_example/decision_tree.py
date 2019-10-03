# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 15:47:20 2019

@author: shansaxena
"""

# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation



# load dataset
pima = pd.read_csv('diabetes.csv')

#split dataset in features and target variable
feature_cols = ['Pregnancies', 'Insulin', 'BMI', 'Age','Glucose','BloodPressure','DiabetesPedigreeFunction','SkinThickness']
X = pima[feature_cols] # Features
y = pima['Outcome'] # Target variable


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred_gini = clf.predict(X_test)


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
"""Accuracy: 0.696969696969697"""


from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('decision_tree.png')
Image(graph.create_png())



"""" entropy criterion"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

clf = DecisionTreeClassifier(criterion = 'entropy', max_depth = 3, min_samples_split = 90)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)
clf.score(X_train,y_train)

#Predict the response for test dataset
y_pred_entropy = clf.predict(X_test)


print("Accuracy:",metrics.accuracy_score(y_test, y_pred_entropy))
"""Accuracy: 0.7705627705627706"""


from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('decision_tree_entropy.png')
Image(graph.create_png())



#tuning thehyperparameters
from sklearn.model_selection import GridSearchCV

parameters={'min_samples_split' : range(10,500,20),'max_depth': range(1,20,2)}

clf_1 = DecisionTreeClassifier()
clf_1=GridSearchCV(clf,parameters)
clf_1 = clf_1.fit(X_train, y_train)

print(clf_1.best_params_)


#precision and recall
from sklearn.metrics import classification_report
target_names = ['class 0', 'class 1']
print(classification_report(y_test, y_pred_entropy, target_names=target_names))


