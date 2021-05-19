from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfTransformer
import csv
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# def save_model(classifier,name_model):
#     with open(name_model+'.model', 'wb') as picklefile:
#         pickle.dump(classifier, picklefile)
#
# def open_model():
#     with open('text_classifier.model', 'rb') as training_model:
#         return pickle.load(training_model)
#
# def learning_model(df):
#     for tag in ['Объектные', 'Функциональные', 'Процессные', 'Ограничения', 'Структурные']:
#         X_train, X_test, y_train, y_test = train_test_split(df['Документ'], df[tag].values,
#                                                             train_size=0.1, random_state=50)
#         count_vect = CountVectorizer()
#         X_train_counts = count_vect.fit_transform(X_train)
#         tfidf_transformer = TfidfTransformer()
#         X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
#
#         clf = RandomForestClassifier(n_estimators=500, random_state=50, n_jobs=-1)
#         clf = clf.fit(X_train_tfidf, y_train)
#         # Тестирование
#         # res_pred = clf.predict(count_vect.transform(X_test))
#         # score = accuracy_score(y_test, res_pred)
#         # print(score)
#         save_model(clf,tag)

def test_m():
    df = pd.read_csv('..\\data\\train_data.csv', encoding='utf-8')
    df.head()
    col = ['Документ', 'Объектные', 'Функциональные', 'Процессные', 'Ограничения', 'Структурные']
    df = df[col]


    test_data = pd.read_csv('..\\data\\test_data.csv', encoding='utf-8')
    col = ['file_id', 'Документ']
    test_data = test_data[col]


    result = {}
    for tag in ['Объектные', 'Функциональные', 'Процессные', 'Ограничения', 'Структурные']:
        X_train, X_test, y_train, y_test = train_test_split(df['Документ'], df[tag].values,
                                                            train_size = 1, random_state=50)
        count_vect = CountVectorizer()
        X_train_counts = count_vect.fit_transform(X_train)
        tfidf_transformer = TfidfTransformer()
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

        clf = RandomForestClassifier(n_estimators=200, random_state=50)
        clf = clf.fit(X_train_tfidf, y_train)
        # res_pred = clf.predict(count_vect.transform(X_test))
        # score = accuracy_score(y_test, res_pred)
        # print(score)
        temp_result = []
        for data in test_data[['Документ']].values:
            temp_result.extend(clf.predict(count_vect.transform(data)))
        result.update({tag: temp_result})

    new_df = {}
    i = 0
    while i < test_data.index.stop:
        str_result = str(i) + "," + str(test_data[['file_id']].values[i][0]) + ",\"" + result.get('Объектные')[i] + "\",\"" \
                     + result.get('Функциональные')[i] + "\",\"" + result.get('Процессные')[i] + "\",\"" \
                     + result.get('Ограничения')[i] + "\",\"" + result.get('Структурные')[i]+"\""
        new_df.update({i: str_result})
        i = i + 1
    res = pd.Series(data=new_df)
    res.to_csv('..\\data\\submission.csv',
               header=['id,file_id,Объектные,Функциональные,Процессные,Ограничения,Структурные'],
               sep="\t", encoding='utf-8', quoting=csv.QUOTE_NONE, escapechar="",index=False)