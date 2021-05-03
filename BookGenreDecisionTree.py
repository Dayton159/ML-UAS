import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from numpy import asarray
from sklearn import tree
import pydotplus
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
import cProfile

def efficiencyTest():
    datainput = pd.read_csv("book_data.csv", delimiter=";")
    giveWarning = False
    X = datainput[['age','gender','genre']].values

    # Data Preprocessing
    from sklearn import preprocessing

    genderValue = ['male', 'female']
    label_gender = preprocessing.LabelEncoder()
    label_gender.fit(genderValue)
    for check in X[:, 1] :
        if check not in genderValue :
            giveWarning = True

    if giveWarning :
        print("Ada kesalahan di penamaan gender")
        return False
    else:
        X[:, 1] = label_gender.transform(X[:, 1])


    genreValue = ['Fantasy', 'Adventure', 'Romance', 'Contemporary' , 'Dystopian', 'Mystery', 'Horror', 'Thriller', 'Paranormal', 'Historical fiction', 'Science Fiction', 'Memoir', 'Cooking', 'Art', 'Self-help / Personal', 'Development', 'Motivational', 'Health', 'History', 'Travel', 'Guide / How-to', 'Families & Relationships', 'Humor', 'Children’s']
    label_genre = preprocessing.LabelEncoder()
    label_genre.fit(genreValue)
    for check in X[:, 2] :
        if check not in genreValue :
            giveWarning = True

    if giveWarning :
        print("Ada kesalahan di penamaan genre")
        return False
    else:

        X[:, 2] = label_genre.transform(X[:, 2])

    y = datainput["suggested"]

    # train_test_split
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

    drugTree = DecisionTreeClassifier(criterion="entropy", max_depth=4)

    drugTree.fit(X_train, y_train)
    predicted = drugTree.predict(X_test)

    print(predicted)

    print(metrics.classification_report(y_test, predicted))

    print("\nDecisionTrees's Accuracy: ", metrics.accuracy_score(y_test, predicted))
    # precision tp / (tp + fp)
    print("\n DecisionTrees's Precision: ", metrics.precision_score(y_test, predicted, average="macro"))
    # recall: tp / (tp + fn)
    recall = metrics.recall_score(y_test, predicted, average="macro")
    print("\n DecisionTree's Recall: ", recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = metrics.f1_score(y_test, predicted, average="macro")
    print("\n DecisionTree's F1: ", f1)

    from sklearn.preprocessing import MinMaxScaler
    scalar = MinMaxScaler()

    normArray = scalar.fit_transform(asarray(X))
    normDataInput = pd.DataFrame(normArray, columns= datainput[['age','gender','genre']].columns)
    print("Model Scalability: \n",normDataInput.head())

    # Show Image
    data = tree.export_graphviz(drugTree, out_file=None, feature_names=['age','gender','genre'])
    graph = pydotplus.graph_from_dot_data(data)
    graph.write_png('mydecisiontree.png')

    img = pltimg.imread('mydecisiontree.png')
    plt.imshow(img)
    plt.show()

cProfile.run('efficiencyTest()')



