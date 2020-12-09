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

# min values
dev_min_lambda = 0
dev_min_perplexity = 0

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
ho_dev_inverse_dic_train = {}

# test data
all_test_words_dict = {}
all_test_words_list = []
# lidstone data
ls_test_words_dict_train = {}
ls_test_words_list_train = []
ls_test_words_dict_test = {}
ls_test_words_list_test = []
# held out data
ho_test_words_dict_train = {}
ho_test_words_list_train = []
ho_test_words_dict_test = {}
ho_test_words_list_test = []
ho_test_inverse_dic_test = {}
ho_test_inverse_dic_train = {}


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


# data preprocessing step
def data_preprocessing_step():
    # lidstone and held out dev data pre processing
    dev_data_preprocessing()

    # lidstone and held out test data pre processing
    test_data_preprocessing()

    # write outputs
    with open(output_filename, 'a+') as output_file:
        output_file.write(generate_output_line(7, len(all_dev_words_list)))


# lidstone model training
def lidstone_model_training():
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
        output_file.write(generate_output_line(16, ls_calculate_perplexity(1 / 100, ls_dev_words_dict_train,
                                                                           ls_dev_words_list_train,
                                                                           ls_dev_words_list_test)))
        output_file.write(generate_output_line(17, ls_calculate_perplexity(1 / 10, ls_dev_words_dict_train,
                                                                           ls_dev_words_list_train,
                                                                           ls_dev_words_list_test)))
        output_file.write(generate_output_line(18, ls_calculate_perplexity(1.00, ls_dev_words_dict_train,
                                                                           ls_dev_words_list_train,
                                                                           ls_dev_words_list_test)))
        output_file.write(generate_output_line(19, dev_min_lambda))
        output_file.write(generate_output_line(20, dev_min_perplexity))


# held out model training
def held_out_model_training():
    # write outputs
    with open(output_filename, 'a+') as output_file:
        output_file.write(generate_output_line(21, len(ho_dev_words_list_train)))
        output_file.write(generate_output_line(22, len(ho_dev_words_list_test)))
        output_file.write(generate_output_line(23,
                                               tr_divided_by_nr(ho_dev_words_dict_train.get(input_word, 0),
                                                                ho_dev_words_dict_train,
                                                                ho_dev_words_dict_test,
                                                                ho_dev_inverse_dic_train)
                                               / len(ho_dev_words_list_train)))
        output_file.write(generate_output_line(24,
                                               tr_divided_by_nr(ho_dev_words_dict_train.get('unseen-word', 0),
                                                                ho_dev_words_dict_train,
                                                                ho_dev_words_dict_test,
                                                                ho_dev_inverse_dic_train)
                                               / len(ho_dev_words_list_test)))


# debug held out model
def debug_ho(train_dict, train_list, test_dict, test_list, train_inverse_dict):
    # held out debug calculation
    tr_divided_by_nr_sum = sum([(tr_divided_by_nr(r, train_dict, test_dict, train_inverse_dict)
                                 / len(test_list)) * nr(r, train_dict, train_inverse_dict) for r in train_inverse_dict.keys()])
    t0_divided_by_n0 = (tr_divided_by_nr(0, train_dict, test_dict, train_inverse_dict)
                        / len(test_list)) * nr(0, train_dict, train_inverse_dict)
    ho_debug_result = tr_divided_by_nr_sum + t0_divided_by_n0

    # write outputs
    with open(output_filename, 'a+') as output_file:
        output_file.write('debug held out: ' + str(ho_debug_result) + '\n')


# model test set evaluation
def model_test_set_evaluation():
    # write outputs
    with open(output_filename, 'a+') as output_file:
        output_file.write(generate_output_line(25, len(all_test_words_list)))
        lidstone_perplexity = ls_calculate_perplexity(dev_min_lambda,
                                                      ls_dev_words_dict_train,
                                                      ls_dev_words_list_train,
                                                      all_test_words_list)
        output_file.write(generate_output_line(26, lidstone_perplexity))
        held_out_perplexity = ho_calculate_perplexity(ho_dev_words_dict_train,
                                                      ho_dev_words_list_train,
                                                      all_test_words_dict,
                                                      all_test_words_list,
                                                      ho_dev_inverse_dic_train)
        output_file.write(generate_output_line(27, held_out_perplexity))
        # output_file.write(generate_output_line(28, 'L' if lidstone_perplexity < held_out_perplexity else 'H'))


# data pre processing
def dev_data_preprocessing():
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
    global ho_dev_inverse_dic_train

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

        # generate inverse dictionary
        ho_dev_inverse_dic_train = inverse_dic(ho_dev_words_dict_train)

        # calculate min perplexity
        global dev_min_lambda
        global dev_min_perplexity
        dev_min_lambda, dev_min_perplexity = ls_find_min_perplexity(ls_dev_words_dict_train,
                                                                    ls_dev_words_list_train,
                                                                    ls_dev_words_list_test)


# test data pre processing
def test_data_preprocessing():
    # data
    global all_test_words_dict
    global all_test_words_list

    # lidstone data
    global ls_test_words_list_train
    global ls_test_words_list_test
    global ls_test_words_dict_train
    global ls_test_words_dict_test

    # held out data
    global ho_test_words_list_train
    global ho_test_words_list_test
    global ho_test_words_dict_train
    global ho_test_words_dict_test
    global ho_test_inverse_dic_test
    global ho_test_inverse_dic_train

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

        train_len = round(len(all_test_words_list) * 0.9)

        # train
        ls_test_words_list_train = all_test_words_list[:train_len]
        for train_token in ls_test_words_list_train:
            ls_test_words_dict_train[train_token] = ls_test_words_dict_train.get(train_token, 0) + 1

        # test
        ls_test_words_list_test = all_test_words_list[train_len:]
        for test_token in ls_test_words_list_test:
            ls_test_words_dict_test[test_token] = ls_test_words_dict_test.get(test_token, 0) + 1

        train_len = round(len(all_test_words_list) * 0.5)

        # train
        ho_test_words_list_train = all_test_words_list[:train_len]
        for train_token in ho_test_words_list_train:
            ho_test_words_dict_train[train_token] = ho_test_words_dict_train.get(train_token, 0) + 1

        # test
        ho_test_words_list_test = all_test_words_list[train_len:]
        for test_token in ho_test_words_list_test:
            ho_test_words_dict_test[test_token] = ho_test_words_dict_test.get(test_token, 0) + 1

        # generate inverse dictionary
        ho_test_inverse_dic_train = inverse_dic(ho_dev_words_dict_train)
        ho_test_inverse_dic_test = inverse_dic(ho_dev_words_dict_test)


# helpers
def tr(r, train_dict, test_dict, train_inverse_dict):
    if r == 0:
        return sum([test_dict.get(word, 0) for word in test_dict.keys() if word not in train_dict.keys()])
    else:
        return sum(test_dict.get(word, 0) for word in train_inverse_dict[r])


def nr(r, train_dict, train_inverse_dict):
    if r == 0:
        return VOCABULARY_SIZE - len(train_dict.keys())
    else:
        return len(train_inverse_dict[r])


def tr_divided_by_nr(r, train_dict, test_dict, train_inverse_dict):
    return tr(r, train_dict, test_dict, train_inverse_dict) / nr(r, train_dict, train_inverse_dict)


def inverse_dic(dict):
    new_dic = {}
    for k, v in dict.items():
        new_dic.setdefault(v, []).append(k)
    return new_dic


def ls_calculate_perplexity(lamb, train_dict, train_list, test_list):
    log_sum = sum([math.log2(ls_word_prob(word, lamb, train_dict, train_list)) for word in test_list])
    return math.pow(2, ((-1 / len(test_list)) * log_sum))


def ls_word_prob(word, lamb, train_dict, train_list):
    # mu = (len(train_list)) / (len(train_list) + (VOCABULARY_SIZE * lamb))
    # mle = train_dict.get(word, 0) / len(train_list)
    # uni = 1 / VOCABULARY_SIZE
    # return (mu * mle) + ((1 - mu) * uni)
    return (train_dict.get(word, 0) + lamb) / (len(train_list) + (VOCABULARY_SIZE * lamb))


def ls_find_min_perplexity(train_dict, train_list, test_list):
    perplexities = [(round(x * 0.01, 2), ls_calculate_perplexity(round(x * 0.01, 2), train_dict, train_list, test_list)) for x in range(1, 201)]
    return min(perplexities, key=lambda x: x[1])


def ho_calculate_perplexity(train_dict, train_list, test_dict, test_list, train_inverse_dict):
    log_sum = sum([math.log2(ho_word_prob(word, train_dict, train_list, test_dict, test_list, train_inverse_dict)) for word in test_list])
    return math.pow(2, ((-1 / len(test_list)) * log_sum))
    # return math.pow(2, (
    #         -1 * (sum(math.log2(ho_word_prob(word, train_dict, train_list, test_dict, test_list, train_inverse_dict)) * test_dict.get(word, 0) for word in test_dict.keys()) / len(test_list))))


def ho_word_prob(word, train_dict, train_list, test_dict, test_list, train_inverse_dict):
    word_r = train_inverse_dict.get(word, 0)
    return tr_divided_by_nr(word_r, train_dict, test_dict, train_inverse_dict) / len(test_list)


# output helpers
def generate_output_line(number, value):
    return f"{OUTPUT_PREFIX}{number}: {value}\n"


def generate_output_file():
    # init step
    init_step()

    # development set preprocessing step
    data_preprocessing_step()

    # lidstone model training
    lidstone_model_training()

    # held out model training
    held_out_model_training()

    # held out debug
    debug_ho(ho_dev_words_dict_train, ho_dev_words_list_train, ho_dev_words_dict_test, ho_dev_words_list_test, ho_dev_inverse_dic_train)

    # model test set evaluation
    model_test_set_evaluation()

    # generate_matrix()


def generate_matrix():
    with open(output_filename, 'a+') as output_file:
        for r in range(0, 10):
            n_r = nr(r, ho_dev_words_dict_train,
                     ho_dev_inverse_dic_train)
            t_r = tr(r, ho_dev_words_dict_train,
                     ho_dev_words_dict_test,
                     ho_dev_inverse_dic_train)
            f_h = round(tr_divided_by_nr(r, ho_dev_words_dict_train,
                                         ho_dev_words_dict_test,
                                         ho_dev_inverse_dic_train), 5)
            prob = (r + dev_min_lambda) / (len(ho_dev_words_list_train) + (VOCABULARY_SIZE * dev_min_lambda))
            f_l = round(prob * t_r, 5)

            output_file.write('\t'.join([str(r), str(f_l), str(f_h), str(n_r), str(t_r)]) + '\n')


# start
if len(sys.argv) >= 4:
    development_set_filename = sys.argv[1]
    test_set_filename = sys.argv[2]
    input_word = sys.argv[3]
    output_filename = sys.argv[4]

    # generate output file
    generate_output_file()
