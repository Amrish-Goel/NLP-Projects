import os
import string
import re
import operator
import json

stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd",
              'your',
              'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers',
              'herself', 'it',
              "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
              'this',
              'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have',
              'has', 'had',
              'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as',
              'until',
              'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during',
              'before',
              'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
              'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each',
              'few',
              'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
              'very',
              's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're',
              've',
              'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',
              "hadn't", 'hasn',
              "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn',
              "needn't",
              'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn',
              "wouldn't"]

counts = dict()
count_pos_tru = dict()
count_pos_dec = dict()
count_neg_tru = dict()
count_neg_dec = dict()

def remove_punctuations(text):
    translator = str.maketrans("", "", string.punctuation)
    text.translate(translator)  # removing punctuations to check not working
    text.replace('[^\w\s]', '')
    return text

def remove_stop_words(text):
    # text = ' '.join([word for word in text.split() if word not in stop_words])  # removing stop words
    # return text
    words = re.sub("[^\w]", " ",  text).split()
    cleaned_text = ' '.join([w.lower() for w in words if w not in stop_words])
    return cleaned_text

def word_count(str, classify):

    words = str.split()

    for word in words:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1

    if classify == 'positive_deceptive':
        for word in words:
            if word in count_pos_dec:
                count_pos_dec[word] += 1
            else:
                count_pos_dec[word] = 1

    elif classify == 'positive_truthful':
        for word in words:
            if word in count_pos_tru:
                count_pos_tru[word] += 1
            else:
                count_pos_tru[word] = 1

    elif classify == 'negative_deceptive':
        for word in words:
            if word in count_neg_dec:
                count_neg_dec[word] += 1
            else:
                count_neg_dec[word] = 1
    elif classify == 'negative_truthful':
        for word in words:
            if word in count_neg_tru:
                count_neg_tru[word] += 1
            else:
                count_neg_tru[word] = 1


def pre_processing(text, classify):
    text = remove_punctuations(text)
    text = remove_stop_words(text)
    word_count(text, classify)
    return text






def read_training_data(training_data_path):
    training_data_dict = dict()

    polarity_directory_list = next(os.walk(training_data_path))[1]
    for polarity_directory in polarity_directory_list:
        # print polarity_directory
        if polarity_directory.startswith("positive"):
            subclass = "positive"
        elif polarity_directory.startswith("negative"):
            subclass = "negative"
        x = os.path.join(training_data_path, polarity_directory)
        deceptive_truthful_directory_list = next(os.walk(x))[1]
        # print deceptive_truthful_directory_list
        for deceptive_truthful_directory in deceptive_truthful_directory_list:
            if deceptive_truthful_directory.startswith("deceptive"):
                classify = subclass + "_deceptive"
            elif deceptive_truthful_directory.startswith("truthful"):
                classify = subclass + "_truthful"
            y = os.path.join(x, deceptive_truthful_directory)
            folders = next(os.walk(y))[1]
            # print folders
            training_data_dict[classify] = list()
            for fold in folders:
                z = os.path.join(y, fold)
                file_list = next(os.walk(z))[2]
                # print file_list
                for file_no in file_list:
                    file_open = open(os.path.join(z, file_no), "r")
                    text = file_open.read().lower().rstrip('\n')  # making it lowercase
                    data = pre_processing(text, classify)
                    training_data_dict[classify].append(data)
    # print(training_data_dict)
    return training_data_dict


def remove_most_least_frequent_word_list(training_data_dict, most_least_frequent_word_list):
    for label, sentences in training_data_dict.items():
        sentences_list = []
        for sentence in sentences:
            # print(sentence)
            words = re.sub("[^\w]", " ", sentence).split()
            sentence = ([w.lower() for w in words if w not in most_least_frequent_word_list])
            sentences_list.append(sentence)
        training_data_dict[label] = sentences_list
    return training_data_dict


def create_most_least_frequent_list():
    newA = dict(sorted(counts.items(), key=operator.itemgetter(1), reverse=True)[:5])
    most_least_frequent_word_list = list(newA.keys())
    for k, v in counts.items():
        if v == 1:
            most_least_frequent_word_list.append(k)

    # removing most-least frequent words from count dict
    for key in most_least_frequent_word_list:
        if key in counts:
            del counts[key]
        if key in count_neg_tru:
            del count_neg_tru[key]
        if key in count_pos_tru:
            del count_pos_tru[key]
        if key in count_pos_dec:
            del count_pos_dec[key]
        if key in count_neg_dec:
            del count_neg_dec[key]
    return most_least_frequent_word_list

def build_nb_model(training_data_dict):
    jsondict = dict()
    prior_probability_dict = dict()
    prior_probability_dict["pos_dec"] = 0.25
    prior_probability_dict["neg_dec"] = 0.25
    prior_probability_dict["pos_tru"] = 0.25
    prior_probability_dict["neg_tru"] = 0.25

    number_of_unique_words = len(counts)
    number_of_words_pos_dec = sum(count_pos_dec.values())
    number_of_words_pos_tru = sum(count_pos_tru.values())
    number_of_words_neg_tru = sum(count_neg_tru.values())
    number_of_words_neg_dec = sum(count_neg_dec.values())
    jsondict["prior_probability"] = prior_probability_dict
    # jsondict["number_of_unique_words"] = number_of_unique_words

    pos_dec_dict = dict()
    # for key, value in count_pos_dec.items():
    #     pos_dec_dict[key] = (value+1)/(number_of_unique_words+number_of_words_pos_dec)
    # jsondict["pos_dec"] = pos_dec_dict
    #
    neg_dec_dict = dict()
    # for key, value in count_neg_dec.items():
    #     neg_dec_dict[key] = (value+1)/(number_of_unique_words+number_of_words_neg_dec)
    # jsondict["neg_dec"] = neg_dec_dict
    #
    pos_tru_dict = dict()
    # for key, value in count_pos_tru.items():
    #     pos_tru_dict[key] = (value+1)/(number_of_unique_words+number_of_words_pos_tru)
    # jsondict["pos_tru"] = pos_tru_dict
    #
    neg_tru_dict = dict()
    # for key, value in count_neg_tru.items():
    #     neg_tru_dict[key] = (value+1)/(number_of_unique_words+number_of_words_neg_tru)
    # jsondict["neg_tru"] = neg_tru_dict
    # print(jsondict)

    for key, value in counts.items():
        if key in count_pos_dec:
            pos_dec_dict[key] = (count_pos_dec[key] + 1) / (number_of_unique_words + number_of_words_pos_dec)
        else:
            pos_dec_dict[key] = 1 / (number_of_unique_words + number_of_words_pos_dec)

        if key in count_neg_dec:
            neg_dec_dict[key] = (count_neg_dec[key] +1)/(number_of_unique_words + number_of_words_neg_dec)
        else:
            neg_dec_dict[key] = 1 / (number_of_unique_words + number_of_words_neg_dec)

        if key in count_pos_tru:
            pos_tru_dict[key] = (count_pos_tru[key] + 1) / (number_of_unique_words + number_of_words_pos_tru)
        else:
            pos_tru_dict[key] = 1/(number_of_unique_words + number_of_words_pos_tru)

        if key in count_neg_tru:
            neg_tru_dict[key] = (count_neg_tru[key] + 1) / (number_of_words_neg_tru + number_of_unique_words)
        else:
            neg_tru_dict[key] = 1 / (number_of_unique_words + number_of_words_neg_tru)

    jsondict["pos_tru"] = pos_tru_dict
    jsondict["neg_tru"] = neg_tru_dict
    jsondict["pos_dec"] = pos_dec_dict
    jsondict["neg_dec"] = neg_dec_dict


    with open('nbmodel.txt', 'w') as fp:
        json.dump(jsondict, fp, indent=2)
    # print(number_of_unique_words)
    # print(number_of_words_neg_dec)
    # print(number_of_words_neg_tru)
    # print(number_of_words_pos_dec)
    # print(number_of_words_pos_tru)



if __name__ == '__main__':
    path = '/Users/amrish/Documents/NLP/HW1/op_spam_training_data'
    training_data_dict = read_training_data(path)

    most_least_frequent_word_list = []
    most_least_frequent_word_list = create_most_least_frequent_list()


    # print(most_least_frequent_word_list)
    training_data_dict = remove_most_least_frequent_word_list(training_data_dict, most_least_frequent_word_list)
    print(training_data_dict)
    # print(count_neg_tru)
    build_nb_model(training_data_dict)

