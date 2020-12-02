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


# helpers


def generate_output_line(number, value):
    return f"{OUTPUT_PREFIX}{number}: {value}\n"


def generate_output_file():
    # init step
    init_step()

    # development set preprocessing step
    development_set_preprocessing_step()

# init step
def init_step():
    with open(output_filename, 'w') as output_file:
        output_file.write(generate_output_line(1, development_set_filename))
        output_file.write(generate_output_line(2, test_set_filename))
        output_file.write(generate_output_line(3, input_word))
        output_file.write(generate_output_line(4, output_filename))
        output_file.write(generate_output_line(5, VOCABULARY_SIZE))
        input_word_uniform_prob = 1 / VOCABULARY_SIZE
        output_file.write(generate_output_line(6, input_word_uniform_prob))


# development set preprocessing step
def development_set_preprocessing_step():
    read_dev_file()


def read_dev_file():
    with open(development_set_filename, 'r') as development_set_file:
        development_set_file_lines = development_set_file.readlines()
        for i in range(0, len(development_set_file_lines), 4):
            article_data = development_set_file_lines[i:i + 4]
            article_train = article_data[0].strip()
            article = article_data[2].strip()
            tokens = article.split()
            for token in tokens:


if len(sys.argv) >= 4:
    development_set_filename = sys.argv[1]
    test_set_filename = sys.argv[2]
    input_word = sys.argv[3]
    output_filename = sys.argv[4]

    # generate output file
    generate_output_file()