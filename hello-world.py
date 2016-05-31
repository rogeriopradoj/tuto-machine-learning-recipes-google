from sklearn import tree
features = [[140, 1], [130, 1], [150, 0], [170, 0]]
# texture feature
# 1 smooth
# 0 bumpy
labels = [0, 0, 1, 1]
# 0 apple
# 1 orange

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)
print(clf.predict([[160, 0]]))

horsepower = [300, 450, 200, 150]
seats = [2, 2, 8, 9]
features = [[h, s] for h, s in zip(horsepower, seats)]
labels = [0, 0, 1, 1]
# 0 sports-car
# 1 minivan

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)
print(clf.predict([[300, 15]]))
