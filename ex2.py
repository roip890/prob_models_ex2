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

# data
dev_words_dict = {}
dev_words_dict_train = {}
dev_words_list_train = []
dev_words_dict_test = {}
dev_words_list_test = []
ho_dev_words_dict_train = {}
ho_dev_words_list_train = []
ho_dev_words_dict_test = {}
ho_dev_words_list_test = []
ho_inverse_dic_train = {}
ho_t0 = 0

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

def calculatePerplexity(lamb):
    sumProbability = 0
    for x in dev_words_list_test:
        sumProbability += math.log2(
            (dev_words_dict_train.get(x, 0) + lamb) / (len(dev_words_list_train) + VOCABULARY_SIZE * lamb))
    return 2 ** (-1 * (sumProbability / len(dev_words_list_test)))

def findMinPerplexity():
    isAlreadyUpdatePerplexity = 0
    for x in range(1, 201):
        lamb = round(x * 0.01, 2)
        tempPerplexity = calculatePerplexity(lamb)
        if isAlreadyUpdatePerplexity == 0:
           isAlreadyUpdatePerplexity = 1
           minPerplexity = tempPerplexity
           minLamb = lamb
           continue
        minPerplexity = min(minPerplexity, tempPerplexity)
        if tempPerplexity == minPerplexity:
            minLamb = lamb
    return minLamb, minPerplexity


def lidstone_model_training():
    with open(output_filename, 'a+') as output_file:
        output_file.write(generate_output_line(8, len(dev_words_list_train)))
        output_file.write(generate_output_line(9, len(dev_words_list_test)))
        output_file.write(generate_output_line(10, len(dev_words_dict_train)))
        output_file.write(generate_output_line(11, len(dev_words_dict_test)))
        output_file.write(generate_output_line(12, dev_words_dict_train.get(input_word, 0) / len(dev_words_list_train)))
        output_file.write(generate_output_line(13, dev_words_dict_train.get('unseen-word', 0) / len(dev_words_list_train)))
        output_file.write(generate_output_line(14, (dev_words_dict_train.get(input_word, 0) + 0.1) / (len(dev_words_list_train) + VOCABULARY_SIZE * 0.1)))
        output_file.write(generate_output_line(15, (dev_words_dict_train.get('unseen-word', 0) + 0.1) / (len(dev_words_list_train) + VOCABULARY_SIZE * 0.1)))
        output_file.write(generate_output_line(16, calculatePerplexity(0.01)))
        output_file.write(generate_output_line(17, calculatePerplexity(0.10)))
        output_file.write(generate_output_line(18, calculatePerplexity(1.00)))
        minLamb, minPerplexity = findMinPerplexity()
        output_file.write(generate_output_line(19, minLamb))
        output_file.write(generate_output_line(20, minPerplexity))
def calculateT0():
    global ho_t0
    for x in ho_dev_words_dict_test.keys():
        if x not in dev_words_dict_train.keys():
            ho_t0 += ho_dev_words_dict_test.get(x, 0)

def inverseDic(dict):
    new_dic = {}
    for k, v in dict.items():
        new_dic.setdefault(v, []).append(k)
    return new_dic

def trDividedByNr(r):
    if r == 0:
        lenZeroRTrain = VOCABULARY_SIZE - len(ho_dev_words_dict_train.keys())
        return ho_t0 / lenZeroRTrain
    sumTr = 0
    for x in ho_inverse_dic_train[r]:
        sumTr += ho_dev_words_dict_test.get(x, 0)
    return sumTr / len(ho_inverse_dic_train[r])

def debug_ho():
    sum = 0
    for x in ho_dev_words_dict_test.keys():
        sum += trDividedByNr(ho_dev_words_dict_train.get(x, 0)) / len(ho_dev_words_list_test)
    return sum + (ho_t0 * (VOCABULARY_SIZE - len(ho_dev_words_dict_train.keys())))/ (len(ho_dev_words_list_test) * (VOCABULARY_SIZE - len(ho_dev_words_dict_train.keys())))

def held_out_model_training():
    global ho_inverse_dic_train
    with open(output_filename, 'a+') as output_file:
        output_file.write(generate_output_line(21, len(ho_dev_words_list_train)))
        output_file.write(generate_output_line(22, len(ho_dev_words_list_test)))
        ho_inverse_dic_train = inverseDic(ho_dev_words_dict_train)
        calculateT0()
        output_file.write(generate_output_line(23, trDividedByNr(ho_dev_words_dict_train.get(input_word, 0)) / len(ho_dev_words_list_test)))
        output_file.write(generate_output_line(24, trDividedByNr(ho_dev_words_dict_train.get('unseen-word', 0)) / len(ho_dev_words_list_test)))
        output_file.write(generate_output_line(25, debug_ho()))

def generate_dev_words_dict():
    global dev_words_list_train
    global dev_words_list_test
    global dev_words_dict_train
    global dev_words_dict_test
    global ho_dev_words_list_train
    global ho_dev_words_list_test
    global ho_dev_words_dict_train
    global ho_dev_words_dict_test
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

        train_len = round(len(dev_words_list) * 0.5)
        # train
        ho_dev_words_list_train = dev_words_list[:train_len]
        for train_token in ho_dev_words_list_train:
            ho_dev_words_dict_train[train_token] = ho_dev_words_dict_train.get(train_token, 0) + 1

        # test
        ho_dev_words_list_test = dev_words_list[train_len:]
        for test_token in ho_dev_words_list_test:
            ho_dev_words_dict_test[test_token] = ho_dev_words_dict_test.get(test_token, 0) + 1


if len(sys.argv) >= 4:
    development_set_filename = sys.argv[1]
    test_set_filename = sys.argv[2]
    input_word = sys.argv[3]
    output_filename = sys.argv[4]

    # generate output file
    generate_output_file()
