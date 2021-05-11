import mqnli.data_util as du
import mqnli.natural_logic_model as nlm
from mqnli.generate_data import process_data


def test_pair(data_path, premise1, hypothesis1, premise2, hypothesis2):
    data, _, _ = process_data(1.0, data_path)
    high_node_labels = ["sentence_q", "subj_adj", "subj_noun", "subj", "neg", "negp", "v_adv", "v_verb", "v_bar", "vp", "vp_q", "obj_adj", "obj_noun", "obj"]
    premise1 = du.parse_simple_sentence(data,premise1)[0]
    premise2 = du.parse_simple_sentence(data,premise2)[0]
    hypothesis1 = du.parse_simple_sentence(data,hypothesis1)[0]
    hypothesis2 = du.parse_simple_sentence(data,hypothesis2)[0]
    partition = {high_node:set() for high_node in high_node_labels}
    for high_node in high_node_labels:
        partition[high_node].add(nlm.compute_simple_relation_intervention(premise1, hypothesis1, premise2, hypothesis2, high_node))
    return partition

path="data/"
print(test_pair(path,"notevery bad singer doesnot badly hears every good tree", "every bad singer doesnot badly hears every good tree", "notevery bad singer doesnot badly hears every good tree","notevery bad singer doesnot badly hears every good tree"))
