import csv
from collections import Counter


def parse_csv(file_path):
    """Парсит файл формата .csv по указанному пути.
          Входное значение file_path - путь к файлу
          Возвращает список из словарей в которых, ключ - название столбца из файла
    """
    with open(file_path, newline='', encoding="utf8") as csv_file:
        print('File encoding: ' + str(csv_file))
        file_data = csv.reader(csv_file)
        columns_name = next(file_data)
        result_parse = []
        for row in file_data:
            new_dict = dict.fromkeys(columns_name)
            for key, item in zip(new_dict, row):
                new_dict[key] = item
            else:
                result_parse.append(new_dict)

        print("Файл '%s' успешно обработан" % file_path)

    return result_parse


def array_handler(classes_array, result_parse):
    class_names = ['Объектные', 'Функциональные', 'Процессные', 'Ограничения', 'Структурные']
    temp_array = []

    for name in class_names:
        print('!!!' + name + '!!!')
        for el in classes_array:
            str1 = ''
            for obj in result_parse:
                str2 = obj[name]
                if el in str2.split(';'):
                    str1 = str2 + " " + obj['Документ']

            c = Counter(str1.split(' '))

            for key, value in c.items():
                temp_array.append(key + ": " + str(value))

        create_csv(name, '../data/submission.csv')
        create_csv(temp_array, '../data/submission.csv')

    # print(temp_array)




def classes_handler(classes_parse, result_parse):
    class_names = ['Объектные', 'Функциональные', 'Процессные', 'Ограничения', 'Структурные']
    object_classes = []

    for name in class_names:
        for obj in classes_parse:
            object_classes.append(obj[name])

        array_handler(object_classes, result_parse)


def create_csv(data, file_path):
    """Создает новый файл .csv или перезаписывает существующий,
        если файл с таким именем существет.
        Входное значение data - данные для записи в файл в виде списка с списков значений для записи;
        file_path - путь + имя файла для сохранения файла.
    """
    if not file_path.endswith(".csv"):
        file_path += ".csv"

    with open(file_path, 'wt', encoding='utf8') as out_file:
        tsv_writer = csv.writer(out_file, delimiter=',')
        # tsv_writer.writerows(data)
        tsv_writer.writerow(data)
        print("Файл '%s' создан успешно" % file_path)