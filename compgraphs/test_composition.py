from compgraphs.mqnli_logic import negation_signatures as vectorized_neg_signatures
from compgraphs.mqnli_logic import relation_composition as vectorized_relation_composition
from compgraphs.mqnli_logic import quantifier_signatures as vectorized_determiner_signatures
from compgraphs.mqnli_logic import MQNLI_Logic_CompGraph, mqnli_logic_compgraph
from intervention import GraphInput
from datasets.mqnli import MQNLIData

idx2rln = ["independence", "equivalence", "entails", "reverse entails", "contradiction", "alternation", "cover"]
rln2idx = {r: i for i, r in enumerate(idx2rln)}


def strong_composition(signature1, signature2, relation1, relation2):
    #returns the stronger relation of the first relation/signature composed
    #with the second relation signature and vice sersa
    composition1 = relation_composition[(signature1[relation1], signature2[relation2])]
    composition2 = relation_composition[(signature2[relation2], signature1[relation1])]
    if composition1 == "independence":
        return composition2
    if composition2 != "independence" and composition1 != composition2:
        print("This shouldn't happen", composition1, composition2)
    return composition1

#creates MacCartney's join operator
relations = ["equivalence", "entails", "reverse entails", "contradiction", "cover", "alternation", "independence"]
relations2 = ["equivalence", "entails", "reverse entails", "contradiction", "cover", "alternation", "independence"]
relation_composition= dict()
for r in relations:
    for r2 in relations2:
        relation_composition[(r,r2)] = "independence"
for r in relations:
    relation_composition[("equivalence", r)] = r
    relation_composition[(r,"equivalence")] = r
relation_composition[("entails", "entails")] = "entails"
relation_composition[("entails", "contradiction")] = "alternation"
relation_composition[("entails", "alternation")] = "alternation"
relation_composition[("reverse entails", "reverse entails")] = "reverse entails"
relation_composition[("reverse entails", "contradiction")] = "cover"
relation_composition[("reverse entails", "cover")] = "cover"
relation_composition[("contradiction", "entails")] = "cover"
relation_composition[("contradiction", "reverse entails")] = "alternation"
relation_composition[("contradiction", "contradiction")] = "equivalence"
relation_composition[("contradiction", "cover")] = "reverse entails"
relation_composition[("contradiction", "alternation")] = "entails"
relation_composition[("alternation", "reverse entails")] = "alternation"
relation_composition[("alternation", "contradiction")] = "entails"
relation_composition[("alternation", "cover")] = "entails"
relation_composition[("cover", "entails")] = "cover"
relation_composition[("cover", "contradiction")] = "reverse entails"
relation_composition[("cover", "alternation")] = "reverse entails"
#create the signatures for negation
negation_signature = {"equivalence":"equivalence", "entails":"reverse entails", "reverse entails":"entails", "contradiction":"contradiction", "cover":"alternation", "alternation":"cover", "independence":"independence"}
emptystring_signature = {"equivalence":"equivalence", "entails":"entails", "reverse entails":"reverse entails", "contradiction":"contradiction", "cover":"cover", "alternation":"alternation", "independence":"independence"}
compose_contradiction_signature = {r:relation_composition[(r, "contradiction")] for r in relations }
#creates the signatures for determiners
determiner_signatures = dict()
symmetric_relation = {"equivalence":"equivalence", "entails":"reverse entails", "reverse entails":"entails", "contradiction":"contradiction", "cover":"cover", "alternation":"alternation", "independence":"independence"}
determiner_signatures[("some","some")] =(
{"equivalence":"equivalence", "entails":"entails", "reverse entails":"reverse entails", "independence":"independence"},
{"equivalence":"equivalence", "entails":"entails", "reverse entails":"reverse entails", "contradiction":"cover", "cover":"cover", "alternation":"independence", "independence":"independence"}
)
determiner_signatures[("every","every")] =(
{"equivalence":"equivalence", "entails":"reverse entails", "reverse entails":"entails", "independence":"independence"},
{"equivalence":"equivalence", "entails":"entails", "reverse entails":"reverse entails", "contradiction":"alternation", "cover":"independence", "alternation":"alternation", "independence":"independence"}
)
for key in determiner_signatures:
    signature1, signature2 = determiner_signatures[key]
    new_signature = dict()
    for key1 in signature1:
        for key2 in signature2:
            new_signature[(key1, key2)] = strong_composition(signature1, signature2, key1, key2)
    determiner_signatures[key] = new_signature

new_signature = dict()
for relation1 in ["equivalence", "entails", "reverse entails", "independence"]:
    for relation2 in relations:
        if (relation2 == "equivalence" or relation2 == "reverse entails") and relation1 != "independence":
            new_signature[(relation1, relation2)] = "reverse entails"
        else:
            new_signature[(relation1, relation2)] = "independence"
determiner_signatures[("some","every")] = new_signature
determiner_signatures[("some","every")][("entails", "contradiction")] = "alternation"
determiner_signatures[("some","every")][("entails", "alternation")] = "alternation"
determiner_signatures[("some","every")][("equivalence", "alternation")] = "alternation"
determiner_signatures[("some","every")][("equivalence", "contradiction")] = "contradiction"
determiner_signatures[("some","every")][("equivalence", "cover")] = "cover"
determiner_signatures[("some","every")][("reverse entails", "cover")] = "cover"
determiner_signatures[("some","every")][("reverse entails", "contradiction")] = "cover"

new_signature = dict()
for key in determiner_signatures[("some", "every")]:
    new_signature[(symmetric_relation[key[0]], symmetric_relation[key[1]])] = symmetric_relation[determiner_signatures["some", "every"][key]]
determiner_signatures[("every", "some")] = new_signature

def compose_signatures(f,g):
    #takes two signatures and returns a signature
    #that is the result of applying the first and then the second
    h = dict()
    for r in f:
        h[r] = g[f[r]]
    return h

def standard_lexical_merge(x,y):
    #merges nouns, adjective, verbs, or adverbs
    if  x == y:
        return "equivalence"
    if x == "":
        return "reverse entails"
    if y == "":
        return "entails"
    return "independence"



def determiner_merge(determiner1,determiner2):
    #merges determiners
    return determiner_signatures[(determiner1,determiner2)]

def negation_merge(negation1, negation2):
    #merges negation
    relations = ["equivalence", "entails", "reverse entails", "contradiction", "cover", "alternation", "independence"]
    if negation1 == negation2 and not negation2:
        return emptystring_signature
    if negation1 == negation2 and negation2 :
        return negation_signature
    if not negation1:
        return compose_contradiction_signature
    if negation1:
        return compose_signatures(negation_signature, compose_contradiction_signature)

def standard_phrase(relation1, relation2):
    #merges a noun relation with an adjective relation
    #or a verb relation with an adverb relation
    if relation2 == "equivalence":
        return relation1
    return "independence"

def determiner_phrase(signature, relation1, relation2):
    #applies a determiner signature to two relation arguments
    return signature[(relation1,relation2)]

def negation_phrase(negation_signature, relation):
    #applies a negation signature to a relation argument
    return negation_signature[relation]

def conjunction_phrase(conjunction_signature, relation1, relation2):
    #applies a conjunction signature to two relation arguments
    return conjunction_signature[(relation1, relation2)]

def get_label(relation):
    #converts MacCartney's relations to 3 class NLI labels
    if relation in ["cover", "independence", "reverse entails"]:
        return "neutral"
    if relation in ["entails", "equivalence"]:
        return "entailment"
    if relation in ["alternation", "contradiction"]:
        return "contradiction"

def compute_simple_relation(premise, hypothesis):
    #computes the relation between a premise and hypothesis simple sentence
    #leaves
    subject_negation_signature = negation_merge(premise.subject_negation, hypothesis.subject_negation)
    subject_determiner_signature = determiner_merge(premise.natlog_subject_determiner, hypothesis.natlog_subject_determiner)
    subject_noun_relation = standard_lexical_merge(premise.subject_noun,hypothesis.subject_noun)
    subject_adjective_relation = standard_lexical_merge(premise.subject_adjective,hypothesis.subject_adjective)
    verb_negation_signature = negation_merge(premise.verb_negation, hypothesis.verb_negation)
    verb_relation = standard_lexical_merge(premise.verb,hypothesis.verb)
    adverb_relation = standard_lexical_merge(premise.adverb,hypothesis.adverb)
    object_negation_signature = negation_merge(premise.object_negation, hypothesis.object_negation)
    object_determiner_signature = determiner_merge(premise.natlog_object_determiner, hypothesis.natlog_object_determiner)
    object_noun_relation = standard_lexical_merge(premise.object_noun,hypothesis.object_noun)
    object_adjective_relation = standard_lexical_merge(premise.object_adjective,hypothesis.object_adjective)

    #the nodes of the tree
    VP_relation = standard_phrase(adverb_relation, verb_relation)
    object_NP_relation = standard_phrase(object_adjective_relation, object_noun_relation)
    subject_NP_relation = standard_phrase(subject_adjective_relation, subject_noun_relation)
    object_DP_relation = determiner_phrase(object_determiner_signature, object_NP_relation, VP_relation)
    object_negDP_relation = negation_phrase(object_negation_signature, object_DP_relation)
    negverb_relation = negation_phrase(verb_negation_signature, object_negDP_relation)
    subject_DP_relation = determiner_phrase(subject_determiner_signature, subject_NP_relation, negverb_relation)
    subject_NegDP_relation = negation_phrase(subject_negation_signature, subject_DP_relation)
    return subject_NegDP_relation

def test_relation_composition():
    for r_1 in range(7):
        for r_2 in range(7):
            r_1_str = idx2rln[r_1]
            r_2_str = idx2rln[r_2]
            res = vectorized_relation_composition[r_1, r_2]
            res_str = idx2rln[res]
            assert relation_composition[(r_1_str, r_2_str)] == res_str

def test_negation_signatures():
    for p_neg in [False, True]:
        for h_neg in [False, True]:
            sig = negation_merge(p_neg, h_neg)
            p_i = 1 if p_neg else 0
            h_i = 1 if h_neg else 0
            vectorized_sig = vectorized_neg_signatures[p_i*2 + h_i].tolist()
            for k, v in enumerate(vectorized_sig):
                key_str = idx2rln[k]
                val_str = idx2rln[v]
                assert sig[key_str] == val_str


def test_determiner_signatures():
    got_sigs = vectorized_determiner_signatures
    for neg_1, neg_1_bool in enumerate([False, True]):
        for neg_2, neg_2_bool in enumerate([False, True]):
            neg_sig = negation_merge(neg_1_bool, neg_2_bool)
            for det_1, det_1_str in enumerate(["some", "every"]):
                for det_2, det_2_str in enumerate(["some", "every"]):
                    det_sigs = determiner_signatures[(det_1_str, det_2_str)]
                    for rln_1, rln_1_str in enumerate(idx2rln[:4]):
                        for rln_2, rln_2_str in enumerate(idx2rln):
                            got = got_sigs[4*(det_1+neg_1*2) + det_2+neg_2*2,
                                           rln_1*7 + rln_2]
                            got_str = idx2rln[got]

                            exp_str = det_sigs[(rln_1_str, rln_2_str)]
                            exp_str = neg_sig[exp_str]

                            assert got_str == exp_str

def compute_simple_relation_gentest(premise, hypothesis, relations_seen=None):
    #computes the relation between a premise and hypothesis simple sentence
    #leaves
    if relations_seen == None:
        relations_seen = dict()
        relations_seen["subNP"] = set()
        relations_seen["objNP"] = set()
        relations_seen["VP"] = set()
        relations_seen["objDP"] = set()
        relations_seen["negobjDP"] = set()
        relations_seen["subDP"] = set()

    subject_negation_signature = negation_merge(premise.subject_negation, hypothesis.subject_negation)
    subject_determiner_signature = determiner_merge(premise.natlog_subject_determiner, hypothesis.natlog_subject_determiner)
    subject_noun_relation = standard_lexical_merge(premise.subject_noun,hypothesis.subject_noun)
    subject_adjective_relation = standard_lexical_merge(premise.subject_adjective,hypothesis.subject_adjective)
    # relations_seen["subNP"].add((subject_noun_relation,subject_adjective_relation))
    verb_negation_signature = negation_merge(premise.verb_negation, hypothesis.verb_negation)
    verb_relation = standard_lexical_merge(premise.verb,hypothesis.verb)
    adverb_relation = standard_lexical_merge(premise.adverb,hypothesis.adverb)
    # relations_seen["VP"].add((verb_relation,adverb_relation))
    object_negation_signature = negation_merge(premise.object_negation, hypothesis.object_negation)
    object_determiner_signature = determiner_merge(premise.natlog_object_determiner, hypothesis.natlog_object_determiner)
    object_noun_relation = standard_lexical_merge(premise.object_noun,hypothesis.object_noun)
    object_adjective_relation = standard_lexical_merge(premise.object_adjective,hypothesis.object_adjective)
    # relations_seen["objNP"].add((object_noun_relation,object_adjective_relation))

    #the nodes of the tree
    VP_relation = standard_phrase(adverb_relation, verb_relation)
    object_NP_relation = standard_phrase(object_adjective_relation, object_noun_relation)
    subject_NP_relation = standard_phrase(subject_adjective_relation, subject_noun_relation)
    object_DP_relation = determiner_phrase(object_determiner_signature, object_NP_relation, VP_relation)
    # relations_seen["objDP"].add((premise.object_determiner, hypothesis.object_determiner, object_NP_relation, VP_relation))
    object_negDP_relation = negation_phrase(object_negation_signature, object_DP_relation)
    negverb_relation = negation_phrase(verb_negation_signature, object_negDP_relation)
    # relations_seen["negobjDP"].add((premise.verb_negation, hypothesis.verb_negation, object_negDP_relation))
    subject_DP_relation = determiner_phrase(subject_determiner_signature, subject_NP_relation, negverb_relation)
    subject_NegDP_relation = negation_phrase(subject_negation_signature, subject_DP_relation)
    # relations_seen["subDP"].add((premise.subject_determiner, hypothesis.subject_determiner, subject_NP_relation, negverb_relation))
    return subject_NegDP_relation, relations_seen


mqnli_mini_data = MQNLIData("../mqnli_data/mini.train.txt",
                     "../mqnli_data/mini.dev.txt",
                     "../mqnli_data/mini.test.txt")

examples = mqnli_mini_data.dev[:5][0]
labels = mqnli_mini_data.dev[:5][1]
example_5batch = examples.transpose(0,1)

graph = MQNLI_Logic_CompGraph(mqnli_mini_data, list(mqnli_logic_compgraph.keys()))

IDX_Q_S, IDX_A_S, IDX_N_S, IDX_NEG, IDX_ADV, IDX_V, \
        IDX_Q_O, IDX_A_O, IDX_N_O = range(9)

class Example:
    def __init__(self, ex):
        self.subject_negation = ex[IDX_Q_S].item() in {graph.no, graph.notevery}
        self.natlog_subject_determiner = "some" \
            if ex[IDX_Q_S].item() in {graph.some, graph.no} else "every"
        self.subject_noun = ex[IDX_N_S].item()
        self.subject_adjective = ex[IDX_A_S].item()
        self.verb_negation = ex[IDX_NEG].item()
        self.verb = ex[IDX_V].item()
        self.adverb = ex[IDX_ADV].item()
        self.object_negation = ex[IDX_Q_O].item() in {graph.no, graph.notevery}
        self.natlog_object_determiner = "some" \
            if ex[IDX_Q_O].item() in {graph.some, graph.no} else "every"
        self.object_noun = ex[IDX_N_O].item()
        self.object_adjective = ex[IDX_A_O].item()


def test_composition_step_by_step():
    i = GraphInput({"input": example_5batch})
    graph.compute(i)

    exs = [(Example(examples[i, :9]), Example(examples[i, 9:])) for i in range(5)]

    expected = [compute_simple_relation_gentest(p, h)[0] for p, h in exs]
    print(expected)
