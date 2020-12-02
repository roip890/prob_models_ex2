import sys

# app params
development_set_filename = ''
test_set_filename = ''
input_word = ''
output_filename = ''

# consts
OUTPUT_PREFIX = 'OUTPUT'
VOCABULARY_SIZE = 300000

# data
dev_words_dict = {}
dev_words_dict_train = {}
dev_words_list_train = []
dev_words_dict_test = {}
dev_words_list_test = []


def generate_output_line(number, value):
    return f"{OUTPUT_PREFIX}{number}: {value}\n"


def generate_output_file():
    # init step
    init_step()

    # development set preprocessing step
    development_set_preprocessing_step()

    # lidstone model training
    lidstone_model_training()


# init step
def init_step():
    with open(output_filename, 'w+') as output_file:
        output_file.write(generate_output_line(1, development_set_filename))
        output_file.write(generate_output_line(2, test_set_filename))
        output_file.write(generate_output_line(3, input_word))
        output_file.write(generate_output_line(4, output_filename))
        output_file.write(generate_output_line(5, VOCABULARY_SIZE))
        input_word_uniform_prob = 1 / VOCABULARY_SIZE
        output_file.write(generate_output_line(6, input_word_uniform_prob))


# development set preprocessing step
def development_set_preprocessing_step():
    generate_dev_words_dict()
    with open(output_filename, 'a+') as output_file:
        output_file.write(generate_output_line(7, len(dev_words_dict.keys())))


def lidstone_model_training():
    with open(output_filename, 'a+') as output_file:
        output_file.write(generate_output_line(8, len(dev_words_list_train)))
        output_file.write(generate_output_line(9, len(dev_words_list_test)))
        output_file.write(generate_output_line(10, len(dev_words_dict_train)))
        output_file.write(generate_output_line(11, len(dev_words_dict_test)))
        output_file.write(generate_output_line(12, dev_words_dict_train.get(input_word, 0) / len(dev_words_list_train)))
        output_file.write(generate_output_line(13, dev_words_dict_train.get('unseen-word', 0) / len(dev_words_list_train)))


def generate_dev_words_dict():
    global dev_words_list_train
    global dev_words_list_test
    with open(development_set_filename, 'r') as development_set_file:
        dev_words_list = []
        development_set_file_lines = development_set_file.readlines()
        for i in range(0, len(development_set_file_lines), 4):
            article_data = development_set_file_lines[i:i + 4]
            article_train = article_data[0].strip()
            article = article_data[2].strip()
            tokens = article.split()
            for token in tokens:
                dev_words_list.append(token)
                dev_words_dict[token] = dev_words_dict.get(token, 0) + 1
        train_len = round(len(dev_words_list) * 0.9)

        # train
        dev_words_list_train = dev_words_list[:train_len]
        for train_token in dev_words_list_train:
            dev_words_dict_train[train_token] = dev_words_dict_train.get(train_token, 0) + 1

        # test
        dev_words_list_test = dev_words_list[train_len:]
        for test_token in dev_words_list_test:
            dev_words_dict_test[test_token] = dev_words_dict_test.get(test_token, 0) + 1


if len(sys.argv) >= 4:
    development_set_filename = sys.argv[1]
    test_set_filename = sys.argv[2]
    input_word = sys.argv[3]
    output_filename = sys.argv[4]

    # generate output file
    generate_output_file()