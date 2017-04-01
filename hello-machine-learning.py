from sklearn import tree
import pydotplus

smooth = 0
bumpy = 1

apple = 0
orange = 1

features = [(140, smooth), (130, smooth), (150, bumpy), (170, bumpy)]
labels = [apple, apple, orange, orange]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

print clf.predict([(160, bumpy)])
