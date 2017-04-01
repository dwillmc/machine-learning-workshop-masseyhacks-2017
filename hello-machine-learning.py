from sklearn import tree

textureSmooth = 0
textureBumpy = 1

labelApple = 0
labelOrange = 1

features = [(140, textureSmooth), (130, textureSmooth), (150, textureBumpy), (170, textureBumpy)]
labels = [labelApple, labelApple, labelOrange, labelOrange]

classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(features, labels)

print classifier.predict([(160, textureBumpy)])
