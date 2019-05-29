import sys
import json

def emission_probabilty_calculation(e_dict, word_dict):
    emission_probabilty = dict()
    emission_probabilty['emission_states'] = dict()
    for key, val in word_dict.items():
        emission_probabilty['emission_states'][key] = dict()
        for k, v in e_dict.items():
            cnt = word_dict[key][k] if word_dict[key].get(k) is not None else 0
            emission_probabilty['emission_states'][key][k] = cnt/v;
    return emission_probabilty

def read_file(fname):
    doc = []
    with open(fname, 'r', encoding='utf-8') as fp:
        for line in fp:
            line = line.strip()
            doc.append(line)
    return doc

def find_initial_states(doc):
    dict_trans = dict()
    begin_states = dict()
    dict_emission = dict()
    dict_word = dict()
    for each_line in doc:
        flag = 0
        before_tag = None
        words = each_line.split(' ')
        for word in words:
            word_tag = word.rsplit('/',1)
            tag = word_tag[1]
            if flag == 0:
                flag = 1
                begin_states[tag] = begin_states[tag]+1 if begin_states.get(tag) is not None else 1

            if before_tag is not None:
                if dict_trans.get(before_tag) is None:
                    dict_trans[before_tag] = dict()
                dict_trans[before_tag][tag] = dict_trans[before_tag][tag] + 1 if dict_trans[before_tag].get(tag) is not None else 1

            before_tag = tag

            dict_emission[tag] = dict_emission[tag]+1 if dict_emission.get(tag) is not None else 1

            if dict_word.get(word_tag[0]) is None:
                dict_word[word_tag[0]] = dict()
            dict_word[word_tag[0]][tag] = dict_word[word_tag[0]][tag]+1 if dict_word[word_tag[0]].get(tag) is not None else 1

    return begin_states, dict_trans, dict_emission, dict_word

def transition_probability_calculation(begin_states, dict_trans, dict_emission, total_lines):
    transition_probabilty = dict()
    transition_probabilty['start_states'] = dict()
    all_tags = len(dict_emission)
    vocab_len = ((all_tags) * (all_tags-1))/2
    transition_probabilty['transition_states'] = dict()

    for key, val in dict_emission.items():
        counter = begin_states[key] if begin_states.get(key) is not None else 0
        transition_probabilty['start_states'][key] = (counter + 1) / (total_lines + all_tags)
        if transition_probabilty['transition_states'].get(key) is None:
            transition_probabilty['transition_states'][key] = dict()
        for k, v in dict_emission.items():
            cnt = dict_trans[key][k] if dict_trans.get(key) is not None and dict_trans[key].get(k) is not None else 0
            val_sum = sum(dict_trans[key].values()) if dict_trans.get(key) is not None else 0
            transition_probabilty['transition_states'][key][k] = (cnt+1) / (val_sum+vocab_len)

    return transition_probabilty


if __name__ == "__main__":
    output_file = "hmmoutput.txt"
    model_file = "hmmmodel.txt"
    train_file = sys.argv[1]
    document = read_file(train_file)
    start_states, transition_dict, emission_dict, word_dict = find_initial_states(document)
    transition_probabilty = transition_probability_calculation(start_states, transition_dict, emission_dict, len(document))
    emission_probability = emission_probabilty_calculation(emission_dict, word_dict)
    final_dict = {**transition_probabilty, **emission_probability}
    model = json.dumps(final_dict, indent=2)
    fout = open(model_file, "w")
    fout.write(model)
    fout.close()
