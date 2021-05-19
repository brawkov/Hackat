from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn import tree
from sklearn.feature_extraction.text import TfidfVectorizer
import csv
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from io import StringIO
import pandas as pd


def test_m():
    # parse_train = parse_csv("../data/train_data.csv")

    df = pd.read_csv('..\\data\\train_data.csv', encoding='utf-8')
    df.head()
    col = ['Документ', 'Объектные', 'Функциональные', 'Процессные', 'Ограничения', 'Структурные']
    df = df[col]

    test_data = pd.read_csv('..\\data\\test_data.csv', encoding='utf-8')
    col = ['file_id', 'Документ']
    test_data = test_data[col]
    # # TODO если несколько значений то в y_train не разбиваются а читаются одним словом
    # print(df['Объектные'].index.stop)
    # new_df = {}
    # # for cls in df['Объектные']:
    # i = 0
    # while i < df['Объектные'].index.stop:
    #     # print(df['Объектные'].values[i].split(';'))
    #     new_df.update({i: df['Объектные'].values[i].split(';')})
    #     i = i + 1
    # ser = pd.Series(data=new_df)
    # print(ser)

    result = {}
    for tag in ['Объектные', 'Функциональные', 'Процессные', 'Ограничения', 'Структурные']:
        X_train, X_test, y_train, y_test = train_test_split(df['Документ'], df[tag].values,
                                                            random_state=0)
        count_vect = CountVectorizer()
        X_train_counts = count_vect.fit_transform(X_train)
        tfidf_transformer = TfidfTransformer()
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
        # clf = MultinomialNB().fit(X_train_tfidf, y_train)
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(X_train_tfidf, y_train)
        temp_result = []
        for data in test_data[['Документ']].values:
            temp_result.extend(clf.predict(count_vect.transform(data)))
        result.update({tag: temp_result})

    new_df = {}
    i = 0
    while i < test_data.index.stop:
        str_result = str(i) + "," + str(test_data[['file_id']].values[i][0]) + "," + result.get('Объектные')[i] + "," \
                     + result.get('Функциональные')[i] + "," + result.get('Процессные')[i] + "," \
                     + result.get('Ограничения')[i] + "," + result.get('Структурные')[i]
        new_df.update({i: str_result})
        i = i + 1
    res = pd.Series(data=new_df)
    print()
    res.to_csv('..\\data\\submission.csv',
               header=['id,file_id,Объектные,Функциональные,Процессные,Ограничения,Структурные'],
               sep="\t", encoding='utf-8', quoting=csv.QUOTE_NONE, escapechar="", index=False)

    # # list of text documents
    # text = ["The quick brown fox jumped over the lazy dog."]
    # # create the transform
    # vectorizer = CountVectorizer()
    # # tokenize and build vocab
    # vectorizer.fit(text)
    # # summarize
    # print(vectorizer.vocabulary_)
    # # encode document
    # vector = vectorizer.transform(text)
    # # summarize encoded vector
    # print(vector.shape)
    # print(type(vector))
    # print(vector.toarray())

    # # list of text documents
    # text = ["The quick brown fox jumped over the lazy dog.",
    #         "The dog.",
    #         "The fox"]
    # # create the transform
    # vectorizer = TfidfVectorizer()
    # # tokenize and build vocab
    # vectorizer.fit(text)
    # # summarize
    # print(vectorizer.vocabulary_)
    # print(vectorizer.idf_)
    # # encode document
    # vector = vectorizer.transform([text[0]])
    # # summarize encoded vector
    # print(vector.shape)
    # print(vector.toarray())
    #
    # print("Обучение")


    # Y = ["test_1"]
    # clf = MultinomialNB().fit(vector.shape,)
    # docs_new = ['God is love', 'OpenGL on the GPU is fast']
    # count_vect = CountVectorizer()
    # X_new_counts = count_vect.transform(docs_new)
    # X_new_tfidf = vector.transform(X_new_counts)
    # r = clf.predict(X_new_tfidf)
    # print(r)


    # print("Обучение")
    # X = [[0, 0], [1, 1]]
    # Y = ["test_1", "test_2","test_3","test_4","test_5"]
    # clf = tree.DecisionTreeClassifier()
    # clf = clf.fit(vector, Y)
    # r = clf.predict([[0.7, 0.7]])
    # r1 = clf.predict_proba([[2., 2.]])
    # print(r)
    # print(r1)
    # print(tree.plot_tree(clf))