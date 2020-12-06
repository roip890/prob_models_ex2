import sys
import math
# app params
development_set_filename = ''
test_set_filename = ''
input_word = ''
output_filename = ''

# consts
OUTPUT_PREFIX = 'OUTPUT'
VOCABULARY_SIZE = 300000

# dev data
all_dev_words_dict = {}
all_dev_words_list = []
# lidstone data
ls_dev_words_dict_train = {}
ls_dev_words_list_train = []
ls_dev_words_dict_test = {}
ls_dev_words_list_test = []
# held out data
ho_dev_words_dict_train = {}
ho_dev_words_list_train = []
ho_dev_words_dict_test = {}
ho_dev_words_list_test = []
ho_inverse_dic_train = {}
# test data
all_test_words_dict = {}
all_test_words_list = []


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
    # lidstone and held out data data pre processing
    lidstone_and_heldout_data_preprocessing()

    # write outputs
    with open(output_filename, 'a+') as output_file:
        output_file.write(generate_output_line(7, len(all_dev_words_list)))


# lidstone model training
def lidstone_model_training():
    # calculate min perplexity
    min_lambda, min_perplexity = find_min_perplexity()

    # write outputs
    with open(output_filename, 'a+') as output_file:
        output_file.write(generate_output_line(8, len(ls_dev_words_list_test)))
        output_file.write(generate_output_line(9, len(ls_dev_words_list_train)))
        output_file.write(generate_output_line(10, len(ls_dev_words_dict_train.keys())))
        output_file.write(generate_output_line(11, ls_dev_words_dict_train.get(input_word, 0)))
        output_file.write(generate_output_line(12, ls_dev_words_dict_train.get(input_word, 0) / len(ls_dev_words_list_train)))
        output_file.write(generate_output_line(13, ls_dev_words_dict_train.get('unseen-word', 0) / len(ls_dev_words_list_train)))
        output_file.write(generate_output_line(14, (ls_dev_words_dict_train.get(input_word, 0) + 0.1) / (len(ls_dev_words_list_train) + VOCABULARY_SIZE * 0.1)))
        output_file.write(generate_output_line(15, (ls_dev_words_dict_train.get('unseen-word', 0) + 0.1) / (len(ls_dev_words_list_train) + VOCABULARY_SIZE * 0.1)))
        output_file.write(generate_output_line(16, calculate_perplexity(0.01)))
        output_file.write(generate_output_line(17, calculate_perplexity(0.10)))
        output_file.write(generate_output_line(18, calculate_perplexity(1.00)))
        output_file.write(generate_output_line(19, min_lambda))
        output_file.write(generate_output_line(20, min_perplexity))


# held out model training
def held_out_model_training():
    # generate inverse dictionary
    global ho_inverse_dic_train
    ho_inverse_dic_train = inverse_dic(ho_dev_words_dict_train)

    # write outputs
    with open(output_filename, 'a+') as output_file:
        output_file.write(generate_output_line(21, len(ho_dev_words_list_train)))
        output_file.write(generate_output_line(22, len(ho_dev_words_list_test)))
        output_file.write(generate_output_line(23, tr_divided_by_nr(ho_dev_words_dict_train.get(input_word, 0)) / len(ho_dev_words_list_train)))
        output_file.write(generate_output_line(24, tr_divided_by_nr(ho_dev_words_dict_train.get('unseen-word', 0)) / len(ho_dev_words_list_test)))


# debug held out model
def debug_ho():
    # held out debug calculation
    tr_divided_by_nr_sum = sum([(tr_divided_by_nr(r) / len(ho_dev_words_list_test)) * nr(r) for r in ho_inverse_dic_train.keys()])
    t0_divided_by_n0 = (tr_divided_by_nr(0) / len(ho_dev_words_list_test)) * nr(0)
    ho_debug_result = tr_divided_by_nr_sum + t0_divided_by_n0

    # write outputs
    with open(output_filename, 'a+') as output_file:
        output_file.write('debug held out: ' + str(ho_debug_result) + '\n')



# model test set evaluation
def model_test_set_evaluation():
    # test data processing
    test_data_preprocessing()

    # write outputs
    with open(output_filename, 'a+') as output_file:
        output_file.write(generate_output_line(25, len(all_test_words_list)))

# data pre processing
def lidstone_and_heldout_data_preprocessing():
    # data
    global all_dev_words_dict
    global all_dev_words_list

    # lidstone data
    global ls_dev_words_list_train
    global ls_dev_words_list_test
    global ls_dev_words_dict_train
    global ls_dev_words_dict_test

    # held out data
    global ho_dev_words_list_train
    global ho_dev_words_list_test
    global ho_dev_words_dict_train
    global ho_dev_words_dict_test

    with open(development_set_filename, 'r') as development_set_file:
        development_set_file_lines = development_set_file.readlines()
        for i in range(0, len(development_set_file_lines), 4):
            article_data = development_set_file_lines[i:i + 4]
            article_train = article_data[0].strip()
            article = article_data[2].strip()
            tokens = article.split()
            for token in tokens:
                all_dev_words_list.append(token)
                all_dev_words_dict[token] = all_dev_words_dict.get(token, 0) + 1

        train_len = round(len(all_dev_words_list) * 0.9)

        # train
        ls_dev_words_list_train = all_dev_words_list[:train_len]
        for train_token in ls_dev_words_list_train:
            ls_dev_words_dict_train[train_token] = ls_dev_words_dict_train.get(train_token, 0) + 1

        # test
        ls_dev_words_list_test = all_dev_words_list[train_len:]
        for test_token in ls_dev_words_list_test:
            ls_dev_words_dict_test[test_token] = ls_dev_words_dict_test.get(test_token, 0) + 1

        train_len = round(len(all_dev_words_list) * 0.5)

        # train
        ho_dev_words_list_train = all_dev_words_list[:train_len]
        for train_token in ho_dev_words_list_train:
            ho_dev_words_dict_train[train_token] = ho_dev_words_dict_train.get(train_token, 0) + 1

        # test
        ho_dev_words_list_test = all_dev_words_list[train_len:]
        for test_token in ho_dev_words_list_test:
            ho_dev_words_dict_test[test_token] = ho_dev_words_dict_test.get(test_token, 0) + 1


# test data pre processing
def test_data_preprocessing():
    # data
    global all_test_words_dict
    global all_test_words_list

    with open(test_set_filename, 'r') as test_set_file:
        test_set_file_lines = test_set_file.readlines()
        for i in range(0, len(test_set_file_lines), 4):
            article_data = test_set_file_lines[i:i + 4]
            article_train = article_data[0].strip()
            article = article_data[2].strip()
            tokens = article.split()
            for token in tokens:
                all_test_words_list.append(token)
                all_test_words_dict[token] = all_test_words_dict.get(token, 0) + 1


# helpers
def tr(r):
    if r == 0:
        return sum([ho_dev_words_dict_test.get(word, 0) for word in ho_dev_words_dict_test.keys() if word not in ho_dev_words_dict_train.keys()])
    else:
        return sum(ho_dev_words_dict_test.get(word, 0) for word in ho_inverse_dic_train[r])


def nr(r):
    if r == 0:
        return VOCABULARY_SIZE - len(ho_dev_words_dict_train.keys())
    else:
        return len(ho_inverse_dic_train[r])


def tr_divided_by_nr(r):
    return tr(r) / nr(r)


def inverse_dic(dict):
    new_dic = {}
    for k, v in dict.items():
        new_dic.setdefault(v, []).append(k)
    return new_dic


def calculate_perplexity(lamb):
    return math.pow(2, (-1 * (sum(math.log2(ls_word_prob(word, lamb)) for word in ls_dev_words_list_test) / len(ls_dev_words_list_test))))


def ls_word_prob(word, lamb):
    return (ls_dev_words_dict_train.get(word, 0) + lamb) / (len(ls_dev_words_list_train) + VOCABULARY_SIZE * lamb)


def find_min_perplexity():
    perplexities = [(round(x * 0.01, 2), calculate_perplexity(round(x * 0.01, 2))) for x in range(1, 201)]
    return min(perplexities, key=lambda x: x[1])


# output helpers
def generate_output_line(number, value):
    return f"{OUTPUT_PREFIX}{number}: {value}\n"


def generate_output_file():
    # init step
    init_step()

    # development set preprocessing step
    development_set_preprocessing_step()

    # lidstone model training
    lidstone_model_training()

    # held out model training
    held_out_model_training()

    # held out debug
    debug_ho()

    # model test set evaluation
    model_test_set_evaluation()

# start
if len(sys.argv) >= 4:
    development_set_filename = sys.argv[1]
    test_set_filename = sys.argv[2]
    input_word = sys.argv[3]
    output_filename = sys.argv[4]

    # generate output file
    generate_output_file()
