import csv_parser


train_path = '../data/train_data.csv'
classes_path = '../data/classes.csv'


def main():
    # print(csv_parser.parse_csv(file_path))
    result_parse = csv_parser.parse_csv(train_path)
    # csv_parser.parse_csv(classes_path, result_parse)
    csv_parser.classes_handler(csv_parser.parse_csv(classes_path), result_parse)
    # csv_parser.array_handler(classes_array, result_parse)


if __name__ == '__main__':
    main()
