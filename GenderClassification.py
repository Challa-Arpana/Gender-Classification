from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


#[height, weight, shoe_size]
X = [[181, 80, 44], [177,70,43,], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39], [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]
Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

# Classifiers
clf1 = tree.DecisionTreeClassifier()
clf2 = KNeighborsClassifier(3)
clf3 = SVC()
clf4 = MLPClassifier(max_iter=1000)

# Fitting and predictions
clf1 = clf1.fit(X, Y)
prediction1 = clf1.predict([[190, 70, 43]])
accuracy_score_1 = accuracy_score(Y, clf1.predict(X)) * 100
print("Decision Tree Classifier:", prediction1)
print("Accuracy for Decision Tree Classifier:", accuracy_score_1)

clf2 = clf2.fit(X, Y)
prediction2 = clf2.predict([[190, 70, 43]])
accuracy_score_2 = accuracy_score(Y, clf2.predict(X)) * 100
print("K Nearest Neighbors:", prediction2)
print("Accuracy for K Nearest Neighbors:", accuracy_score_2)

clf3 = clf3.fit(X, Y)
prediction3 = clf3.predict([[190, 70, 43]])
accuracy_score_3 = accuracy_score(Y, clf3.predict(X)) * 100
print("SVC:", prediction3)
print("Accuracy for SVC:", accuracy_score_3)

clf4 = clf4.fit(X, Y)
prediction4 = clf4.predict([[190, 70, 43]])
accuracy_score_4 = accuracy_score(Y, clf4.predict(X)) * 100
print("Neural Network:", prediction4)
print("Accuracy for Neural Network:", accuracy_score_4)

# Finding the best classifier
best_accuracy = max(accuracy_score_1, accuracy_score_2, accuracy_score_3, accuracy_score_4)

if best_accuracy == accuracy_score_1:
    best_classifier = "Decision Tree Classifier"
elif best_accuracy == accuracy_score_2:
    best_classifier = "K Nearest Neighbors"
elif best_accuracy == accuracy_score_3:
    best_classifier = "SVC"
else:
    best_classifier = "Neural Network"

print(f"The best classifier is: {best_classifier} with an accuracy of {best_accuracy:.2f}%")
