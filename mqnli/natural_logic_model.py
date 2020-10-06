import json
import copy
import random
from mqnli.data_util import sentence

# copied from MultiplyQuantifiedData repo, for debugging mqnli_logic compgraph

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
relation_composition[("cover", "alternation")] = " reverse entails"
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


#creates the signature for or
and_signature = dict()
for relation1 in relations:
    for relation2 in relations2:
        if relation2 in ["contradiction", "alternation"] or relation1 in ["contradiction", "alternation"]:
            and_signature[(relation1,relation2)] = "alternation"
        else:
            and_signature[(relation1,relation2)] = "independence"
and_signature[("equivalence", "equivalence")] = "equivalence"
and_signature[("equivalence", "entails")] = "entails"
and_signature[("equivalence", "reverse entails")] = "reverse entails"
and_signature[("entails", "equivalence")] = "entails"
and_signature[("entails", "entails")] = "entails"
and_signature[("reverse entails", "equivalence")] = "reverse entails"
and_signature[("reverse entails", "reverse entails")] = "reverse entails"

or_signature = dict()
for relation in relations:
    for relation2 in relations2:
        or_signature[(relation, relation2)] = negation_signature[and_signature[(negation_signature[relation], negation_signature[relation2])]]

if_signature = dict()
for relation in relations:
    for relation2 in relations2:
        if_signature[(relation, relation2)] = or_signature[(negation_signature[relation], relation2)]

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
    relations_seen["subNP"].add((subject_noun_relation,subject_adjective_relation))
    verb_negation_signature = negation_merge(premise.verb_negation, hypothesis.verb_negation)
    verb_relation = standard_lexical_merge(premise.verb,hypothesis.verb)
    adverb_relation = standard_lexical_merge(premise.adverb,hypothesis.adverb)
    relations_seen["VP"].add((verb_relation,adverb_relation))
    object_negation_signature = negation_merge(premise.object_negation, hypothesis.object_negation)
    object_determiner_signature = determiner_merge(premise.natlog_object_determiner, hypothesis.natlog_object_determiner)
    object_noun_relation = standard_lexical_merge(premise.object_noun,hypothesis.object_noun)
    object_adjective_relation = standard_lexical_merge(premise.object_adjective,hypothesis.object_adjective)
    relations_seen["objNP"].add((object_noun_relation,object_adjective_relation))

    #the nodes of the tree
    VP_relation = standard_phrase(adverb_relation, verb_relation)
    object_NP_relation = standard_phrase(object_adjective_relation, object_noun_relation)
    subject_NP_relation = standard_phrase(subject_adjective_relation, subject_noun_relation)
    object_DP_relation = determiner_phrase(object_determiner_signature, object_NP_relation, VP_relation)
    relations_seen["objDP"].add((premise.object_determiner, hypothesis.object_determiner, object_NP_relation, VP_relation))
    object_negDP_relation = negation_phrase(object_negation_signature, object_DP_relation)
    negverb_relation = negation_phrase(verb_negation_signature, object_negDP_relation)
    relations_seen["negobjDP"].add((premise.verb_negation, hypothesis.verb_negation, object_negDP_relation))
    subject_DP_relation = determiner_phrase(subject_determiner_signature, subject_NP_relation, negverb_relation)
    subject_NegDP_relation = negation_phrase(subject_negation_signature, subject_DP_relation)
    relations_seen["subDP"].add((premise.subject_determiner, hypothesis.subject_determiner, subject_NP_relation, negverb_relation))
    return subject_NegDP_relation, relations_seen

def conjunction_to_negation(conjunction):
    if conjunction == "or":
        return False,False,False
    if conjunction == "and":
        return True,True,True
    if conjunction == "then":
        return True,False,False

def compute_boolean_relation(premise_sentence1, premise_conjunction,premise_sentence2, hypothesis_sentence1, hypothesis_conjunction,hypothesis_sentence2):
    #computes the relation between a premise and hypothesis compound sentence
    premise_sentence1_negation, premise_conjunction_negation, premise_sentence2_negation= conjunction_to_negation(premise_conjunction)
    hypothesis_sentence1_negation, hypothesis_conjunction_negation, hypothesis_sentence2_negation= conjunction_to_negation(hypothesis_conjunction)
    sentence1_negation_signature = negation_merge(premise_sentence1_negation,hypothesis_sentence1_negation)
    sentence1_relation = compute_simple_relation(premise_sentence1, hypothesis_sentence1)
    sentence2_negation_signature = negation_merge(premise_sentence2_negation,hypothesis_sentence2_negation)
    sentence2_relation = compute_simple_relation(premise_sentence2, hypothesis_sentence2)
    sentence1_negation_relation = negation_phrase(sentence1_negation_signature, sentence1_relation)
    sentence2_negation_relation = negation_phrase(sentence2_negation_signature, sentence2_relation)
    conjunction_signature = or_signature
    conjunction_relation = conjunction_phrase(conjunction_signature, sentence1_negation_relation, sentence2_negation_relation)
    conjunction_negation_signature = negation_merge(premise_conjunction_negation, hypothesis_conjunction_negation)
    conjunction_negation_relation = negation_phrase(conjunction_negation_signature, conjunction_relation)
    return conjunction_negation_relation

def compute_boolean_relation_test(sentence1_relation,sentence2_relation, premise_conjunction,hypothesis_conjunction):
    #computes the relation between a premise and hypothesis compound sentence
    premise_sentence1_negation, premise_conjunction_negation, premise_sentence2_negation= conjunction_to_negation(premise_conjunction)
    hypothesis_sentence1_negation, hypothesis_conjunction_negation, hypothesis_sentence2_negation= conjunction_to_negation(hypothesis_conjunction)
    sentence1_negation_signature = negation_merge(premise_sentence1_negation,hypothesis_sentence1_negation)
    sentence2_negation_signature = negation_merge(premise_sentence2_negation,hypothesis_sentence2_negation)
    sentence1_negation_relation = negation_phrase(sentence1_negation_signature, sentence1_relation)
    sentence2_negation_relation = negation_phrase(sentence2_negation_signature, sentence2_relation)
    conjunction_signature = or_signature
    conjunction_relation = conjunction_phrase(conjunction_signature, sentence1_negation_relation, sentence2_negation_relation)
    conjunction_negation_signature = negation_merge(premise_conjunction_negation, hypothesis_conjunction_negation)
    conjunction_negation_relation = negation_phrase(conjunction_negation_signature, conjunction_relation)
    return conjunction_negation_relation

def basemod(base, mod, relation):
    if relation == "equivalence":
        return "(" + base + "*" + mod + "+" + base + ")"
    if relation == "entails":
        return "(" +base + "*" + mod + ")"
    if relation == "reverse entails":
        return "(" +base + "*" + mod + ")"
    else:
        return "(" +base + "*" + base + "*" + "(" + "1 + " + mod + ")" + "*" + "(" + "1 + " + mod + ")" + "- 3*"+base + "*" + mod + "-" + base +  ")"

def test_simple():
    placerelations = ["equivalence", "entails", "reverse entails","alternation", "contradiction", "cover",  "independence"]
    conjs = ["or", "and", "then"]
    badbools = []
    for r in relations:
        for r2 in relations2:
            for c1 in ["or", "and", "then"]:
                for c2 in ["or", "and", "then"]:
                    if r == "independence" and get_label(compute_boolean_relation_test(r, r2,c1,c2)) == "neutral" and (get_label(compute_boolean_relation_test("entails", r2,c1,c2)) != "neutral" or get_label(compute_boolean_relation_test("alternation", r2,c1,c2)) != "neutral" or get_label(compute_boolean_relation_test("reverse entails", r2,c1,c2)) != "neutral"):
                        badbools.append((conjs.index(c1),conjs.index(c2),placerelations.index(r),placerelations.index(r2)))
                    if r2 == "independence" and get_label(compute_boolean_relation_test(r, r2,c1,c2)) == "neutral" and (get_label(compute_boolean_relation_test(r,"entails", c1,c2)) != "neutral" or get_label(compute_boolean_relation_test(r,"alternation", c1,c2)) != "neutral" or get_label(compute_boolean_relation_test(r,"reverse entails", c1,c2)) != "neutral"):
                        badbools.append((conjs.index(c1),conjs.index(c2),placerelations.index(r),placerelations.index(r2)))
    print(badbools)
    for x in badbools:
        print(x)



    x = {"neutral":dict(), "contradiction":dict(), "entailment":dict()}
    for k in x:
        for VP_relation in ["equivalence", "entails", "reverse entails", "independence"]:
            for object_NP_relation in ["equivalence", "entails", "reverse entails", "independence"]:
                for subject_NP_relation in ["equivalence", "entails", "reverse entails", "independence"]:
                    x[k][(VP_relation, object_NP_relation, subject_NP_relation)] = 0
    for VP_relation in ["equivalence", "entails", "reverse entails", "independence"]:
        for object_NP_relation in ["equivalence", "entails", "reverse entails", "independence"]:
            for subject_NP_relation in ["equivalence", "entails", "reverse entails", "independence"]:
                for subject_negation_signature in [negation_merge(x, y) for x in [True, False] for y in [True, False]]:
                    for object_negation_signature in [negation_merge(x, y) for x in [True, False] for y in [True, False]]:
                        for verb_negation_signature in [negation_merge(x, y) for x in [True, False] for y in [True, False]]:
                            for subject_determiner_signature in [determiner_merge(x, y) for x in ["every", "some"] for y in ["every", "some"]]:
                                for object_determiner_signature in [determiner_merge(x, y) for x in ["every", "some"] for y in ["every", "some"]]:
                                    object_DP_relation = determiner_phrase(object_determiner_signature, object_NP_relation, VP_relation)
                                    object_negDP_relation = negation_phrase(object_negation_signature, object_DP_relation)
                                    negverb_relation = negation_phrase(verb_negation_signature, object_negDP_relation)
                                    subject_DP_relation = determiner_phrase(subject_determiner_signature, subject_NP_relation, negverb_relation)
                                    subject_NegDP_relation = negation_phrase(subject_negation_signature, subject_DP_relation)
                                    x[get_label(subject_NegDP_relation)][(VP_relation, object_NP_relation, subject_NP_relation)] +=1
    count = 0
    count2 = 0
    for k in x["neutral"]:
        if k[0] != "independence" and k[1] != "independence" and k[2] != "independence":
            count += x["neutral"][k]
        count2 += x["neutral"][k]
    print(count,count2,count/count2)
    expression = ""
    for k in x["entails"]:
        if x["entails"][k] != 0:
            expression += str(x["entails"][k]) + "*" + basemod("v", "r", k[0]) + "*" + basemod("o", "b", k[1]) +"*" + basemod("s", "a", k[2]) + "+"
    print(expression, "\n\n")
    expression = ""
    for k in x["contradicts"]:
        if x["contradicts"][k] != 0:
            expression += str(x["contradicts"][k]) + "*" + basemod("v", "r", k[0]) + "*" + basemod("o", "b", k[1]) +"*" + basemod("s", "a", k[2]) + "+"
    print(expression, "\n\n")
    expression = ""
    for k in x["neutral"]:
        if x["neutral"][k] != 0:
            expression += str(x["neutral"][k]) + "*" + basemod("v", "r", k[0]) + "*" + basemod("o", "b", k[1]) +"*" + basemod("s", "a", k[2]) + "+"
    print(expression, "\n\n")
    expression = ""
    for k in x["entails"]:
        if x["entails"][k] != 0:
            expression += str(x["entails"][k]) + "*" + basemod("50", "50", k[0]) + "*" + basemod("50", "50", k[1]) +"*" + basemod("50", "50", k[2]) + "+"
    print(expression, "\n\n")

def create_gen_split(bigratio):
    dets1 = ["some", "every"]
    dets2 = ["some", "every"]
    negs1 = [True, False]
    negs2 = [True, False]
    subrels = ["entails", "reverse entails", "independence", "equivalence"]
    verbrels = ["entails", "reverse entails", "independence", "equivalence"]
    objrels = ["entails", "reverse entails", "independence", "equivalence"]
    equiv_classes = dict()
    for r in relations:
        equiv_classes[r] = []
    for det1 in dets1:
        for neg1 in negs1:
            for det2 in dets2:
                for neg2 in negs2:
                    for objrel in objrels:
                        for verbrel in verbrels:
                            object_negation_signature = negation_merge(neg1, neg2)
                            object_determiner_signature = determiner_merge(det1, det2)
                            object_DP_relation = determiner_phrase(object_determiner_signature, objrel, verbrel)
                            object_NegDP_relation = negation_phrase(object_negation_signature, object_DP_relation)
                            example = dict()
                            if neg1:
                                if det1 == "some":
                                    example["premobjdet"] = "no"
                                if det1 == "every":
                                    example["premobjdet"] = "notevery"
                            else:
                                example["premobjdet"] = det1
                            if neg2:
                                if det2 == "some":
                                    example["hypobjdet"] = "no"
                                if det2 == "every":
                                    example["hypobjdet"] = "notevery"
                            else:
                                example["hypobjdet"] = det2
                            example["objmod"] = objrel
                            example["verbmod"] = verbrel
                            equiv_classes[object_NegDP_relation].append(example)
    negation_equiv_classes = dict()
    negation_equiv_classes["notnot"] = dict()
    negation_equiv_classes["emptynot"] = dict()
    negation_equiv_classes["notempty"] = dict()
    negation_equiv_classes["emptyempty"] = dict()
    for rel in equiv_classes:
        random.shuffle(equiv_classes[rel])
        shared = equiv_classes[rel][0:int(len(equiv_classes[rel])*bigratio)]
        notshared = equiv_classes[rel][int(len(equiv_classes[rel])*bigratio):]
        num_examples = len(notshared)
        print(rel, num_examples)
        negation_equiv_classes["notnot"][rel] = copy.deepcopy(notshared[0:int(num_examples/4)])
        negation_equiv_classes["emptynot"][rel] = copy.deepcopy(notshared[int(num_examples/4):int((2*num_examples)/4)])
        negation_equiv_classes["notempty"][rel] = copy.deepcopy(notshared[int((2*num_examples)/4):int((3*num_examples)/4)])
        negation_equiv_classes["emptyempty"][rel] = copy.deepcopy(notshared[int((3*num_examples)/4):])
        negation_equiv_classes["notnot"][rel] += copy.deepcopy(shared)
        negation_equiv_classes["emptynot"][rel] += copy.deepcopy(shared)
        negation_equiv_classes["notempty"][rel] += copy.deepcopy(shared)
        negation_equiv_classes["emptyempty"][rel] += copy.deepcopy(shared)
    equiv_classes = dict()
    for r in relations:
        equiv_classes[r] = []
    for neg1 in negs1:
        for neg2 in negs2:
            for rel in relations:
                negation_signature = negation_merge(neg1, neg2)
                negverb_relation = negation_phrase(negation_signature, rel)
                if neg1 and neg2:
                    for example in negation_equiv_classes["notnot"][rel]:
                        example["premnegation"] = neg1
                        example["hypnegation"] = neg2
                        equiv_classes[negverb_relation].append(example)
                if neg1 and not neg2:
                    for example in negation_equiv_classes["notempty"][rel]:
                        example["premnegation"] = neg1
                        example["hypnegation"] = neg2
                        equiv_classes[negverb_relation].append(example)
                if not neg1 and not neg2:
                    for example in negation_equiv_classes["emptyempty"][rel]:
                        example["premnegation"] = neg1
                        example["hypnegation"] = neg2
                        equiv_classes[negverb_relation].append(example)
                if not neg1 and neg2:
                    for example in negation_equiv_classes["emptynot"][rel]:
                        example["premnegation"] = neg1
                        example["hypnegation"] = neg2
                        equiv_classes[negverb_relation].append(example)
    det_equiv_classes = dict()
    for det1 in dets1:
        for neg1 in negs1:
            for det2 in dets2:
                for neg2 in negs2:
                    for subrel in subrels:
                        det_equiv_classes[(det1,neg1,det2,neg2,subrel)] = dict()
    for rel in equiv_classes:
        i = 0
        random.shuffle(equiv_classes[rel])
        shared = equiv_classes[rel][0:int(len(equiv_classes[rel])*bigratio)]
        notshared = equiv_classes[rel][int(len(equiv_classes[rel])*bigratio):]
        num_examples = len(notshared)
        print(rel, num_examples)
        for neg2 in negs2:
            for det1 in dets1:
                for det2 in dets2:
                    for subrel in subrels:
                        for neg1 in negs1:
                            if num_examples > 64:
                                det_equiv_classes[(det1,neg1,det2,neg2,subrel)][rel] = copy.deepcopy(notshared[int((i*num_examples)/64):int(((i+1)*num_examples)/64)])
                            else:
                                det_equiv_classes[(det1,neg1,det2,neg2,subrel)][rel] = [copy.deepcopy(notshared[i%num_examples])]
                                if i % num_examples == num_examples - 1:
                                    random.shuffle(notshared)
                            i += 1
                            det_equiv_classes[(det1,neg1,det2,neg2,subrel)][rel] += copy.deepcopy(shared)
    final_result = []
    for det1 in dets1:
        for neg1 in negs1:
            for det2 in dets2:
                for neg2 in negs2:
                    for subrel in subrels:
                        for rel in relations:
                            subject_negation_signature = negation_merge(neg1, neg2)
                            subject_determiner_signature = determiner_merge(det1, det2)
                            subject_DP_relation = determiner_phrase(subject_determiner_signature, subrel, rel)
                            subject_NegDP_relation = negation_phrase(subject_negation_signature, subject_DP_relation)
                            for example in det_equiv_classes[(det1,neg1,det2,neg2,subrel)][rel]:
                                if neg1:
                                    if det1 == "some":
                                        example["premsubdet"] = "no"
                                    if det1 == "every":
                                        example["premsubdet"] = "notevery"
                                else:
                                    example["premsubdet"] = det1
                                if neg2:
                                    if det2 == "some":
                                        example["hypsubdet"] = "no"
                                    if det2 == "every":
                                        example["hypsubdet"] = "notevery"
                                else:
                                    example["hypsubdet"] = det2
                                example["submod"] = subrel
                                example["negverb"] = rel
                                example["extra"] = subject_NegDP_relation
                                final_result.append(example)
    total = len(final_result)
    print(total)
    dets1 = ["some", "every", "no","notevery"]
    dets2 = ["some", "every", "no","notevery"]
    verbfix = dict()
    i = 0
    options = [1,2,3,4,5]
    random.shuffle(options)
    shared = options[0:int(5*bigratio)]
    notshared = options[int(5*bigratio):]
    random.shuffle(notshared)
    for det1 in dets1:
        for det2 in dets2:
            for objrel in objrels:
                verbfix[(det1,det2,objrel)] = [notshared[i%len(notshared)]]
                verbfix[(det1,det2,objrel)] += copy.deepcopy(shared)
                if i%len(notshared) == len(notshared) - 1:
                    random.shuffle(notshared)
                i+= 1
    objfix = dict()
    i = 0
    random.shuffle(options)
    shared = options[0:int(5*bigratio)]
    notshared = options[int(5*bigratio):]
    random.shuffle(notshared)
    for det1 in dets1:
        for det2 in dets2:
            for verbrel in verbrels:
                objfix[(det1,det2,verbrel)] = [notshared[i%len(notshared)]]
                objfix[(det1,det2,verbrel)]+= copy.deepcopy(shared)
                if i%len(notshared) == len(notshared) - 1:
                    random.shuffle(notshared)
                i += 1
    subfix = dict()
    i = 0
    random.shuffle(options)
    shared = options[0:int(5*bigratio)]
    notshared = options[int(5*bigratio):]
    random.shuffle(notshared)
    spread = {"neutral":0, "contradiction":0, "entailment":0}
    for det1 in dets1:
        for det2 in dets2:
            for rel in relations:
                subfix[(det1,det2,rel)] = [notshared[i%len(notshared)]]
                subfix[(det1,det2,rel)] += copy.deepcopy(shared)
                if i%len(notshared) == len(notshared) - 1:
                    random.shuffle(notshared)
                i += 1
    print(shared, notshared)
    true_final_result = []
    memes = set()
    for i in range(total):
        memes.add((final_result[i]["premsubdet"], final_result[i]["hypsubdet"], final_result[i]["negverb"]))
        temp = []
        if final_result[i]["objmod"] == "independence":
            type = objfix[(final_result[i]["premobjdet"], final_result[i]["hypobjdet"], final_result[i]["verbmod"])]
            if 1 in type:
                final_result[i]["objmod"] = "entails"
                final_result[i]["obj"] = "independence"
                temp.append(copy.deepcopy(final_result[i]))
            if 2 in type:
                final_result[i]["objmod"] = "reverse entails"
                final_result[i]["obj"] = "independence"
                temp.append(copy.deepcopy(final_result[i]))
            if 3 in type:
                final_result[i]["objmod"] = "equivalence"
                final_result[i]["obj"] = "independence"
                temp.append(copy.deepcopy(final_result[i]))
            if 4 in type:
                final_result[i]["objmod"] = "independence"
                final_result[i]["obj"] = "independence"
                temp.append(copy.deepcopy(final_result[i]))
            if 5 in type:
                final_result[i]["objmod"] = "independence"
                final_result[i]["obj"] = "equivalence"
                temp.append(copy.deepcopy(final_result[i]))
        else:
            final_result[i]["obj"] = "equivalence"
            temp.append(copy.deepcopy(final_result[i]))
        temp2 = []
        if final_result[i]["verbmod"] == "independence":
            type = verbfix[(final_result[i]["premobjdet"], final_result[i]["hypobjdet"], final_result[i]["objmod"])]
            if 1 in type:
                for example in temp:
                    example["verbmod"] = "entails"
                    example["verb"] = "independence"
                    temp2.append(copy.deepcopy(example))
            if 2 in type:
                for example in temp:
                    example["verbmod"] = "reverse entails"
                    example["verb"] = "independence"
                    temp2.append(copy.deepcopy(example))
            if 3 in type:
                for example in temp:
                    example["verbmod"] = "equivalence"
                    example["verb"] = "independence"
                    temp2.append(copy.deepcopy(example))
            if 4 in type:
                for example in temp:
                    example["verbmod"] = "independence"
                    example["verb"] = "independence"
                    temp2.append(copy.deepcopy(example))
            if 5 in type:
                for example in temp:
                    example["verbmod"] = "independence"
                    example["verb"] = "equivalence"
                    temp2.append(copy.deepcopy(example))
        else:
            for example in temp:
                example["verbmod"] = final_result[i]["verbmod"]
                example["verb"] = "equivalence"
                temp2.append(copy.deepcopy(example))
        temp3 = []
        if final_result[i]["submod"] == "independence":
            type = subfix[(final_result[i]["premsubdet"], final_result[i]["hypsubdet"], final_result[i]["negverb"])]
            if 1 in type:
                for example in temp2:
                    example["submod"] = "entails"
                    example["sub"] = "independence"
                    temp3.append(copy.deepcopy(example))
            if 2 in type:
                for example in temp2:
                    example["submod"] = "reverse entails"
                    example["sub"] = "independence"
                    temp3.append(copy.deepcopy(example))
            if 3 in type:
                for example in temp2:
                    example["submod"] = "equivalence"
                    example["sub"] = "independence"
                    temp3.append(copy.deepcopy(example))
            if 4 in type:
                for example in temp2:
                    example["submod"] = "independence"
                    example["sub"] = "independence"
                    temp3.append(copy.deepcopy(example))
            if 5 in type:
                for example in temp2:
                    example["submod"] = "independence"
                    example["sub"] = "equivalence"
                    temp3.append(copy.deepcopy(example))
        else:
            for example in temp2:
                example["submod"] = final_result[i]["submod"]
                example["sub"] = "equivalence"
                temp3.append(copy.deepcopy(example))
        true_final_result += temp3
        spread[get_label(final_result[i].pop("extra", None))] += 1
    print(len(memes), len(memes))
    final_result = true_final_result
    final_encodings = set()
    easycount = 0
    memes = set()
    for example in final_result:
        memes.add((example["submod"], example["sub"]))
        if "verb" not in example:
            print(example)
        if example["verb"] == "independence" or example["verbmod"] == "independence" or example["sub"] == "independence" or example["submod"] == "independence" or example["obj"] == "independence" or example["objmod"] == "independence":
            easycount +=1
        encoding = []
        dets = ["every", "notevery", "some", "no"]
        if example["premnegation"]:
            encoding.append(1)
        else:
            encoding.append(0)
        encoding += [dets.index(example["premsubdet"]),dets.index(example["premobjdet"])]
        if example["hypnegation"]:
            encoding.append(1)
        else:
            encoding.append(0)
        encoding += [dets.index(example["hypsubdet"]),dets.index(example["hypobjdet"])]
        if example["submod"] == "equivalence":
            encoding.append(0)
        if example["submod"] == "reverse entails":
            encoding.append(1)
        if example["submod"] == "entails":
            encoding.append(2)
        if example["submod"] == "independence":
            encoding.append(3)
        if example["objmod"] == "equivalence":
            encoding.append(0)
        if example["objmod"] == "reverse entails":
            encoding.append(1)
        if example["objmod"] == "entails":
            encoding.append(2)
        if example["objmod"] == "independence":
            encoding.append(3)
        if example["verbmod"] == "equivalence":
            encoding.append(0)
        if example["verbmod"] == "reverse entails":
            encoding.append(1)
        if example["verbmod"] == "entails":
            encoding.append(2)
        if example["verbmod"] == "independence":
            encoding.append(3)
        if example["sub"] == "equivalence":
            encoding.append(1)
        else:
            encoding.append(0)
        if example["verb"] == "equivalence":
            encoding.append(1)
        else:
            encoding.append(0)
        if example["obj"] == "equivalence":
            encoding.append(1)
        else:
            encoding.append(0)
        if len(encoding) != 12:
            print("oh fuck")
        final_encodings.add(json.dumps(encoding))
    print(total, 16*4*4*4*4*16)
    print(spread, easycount)
    inverse_encodings = set()
    for a in range(2):
        for b in range(4):
            for c in range(4):
                for d in range(2):
                    for e in range(4):
                        for f in range(4):
                            for g in range(4):
                                for h in range(4):
                                    for i in range(4):
                                        for j in range(2):
                                            for k in range(2):
                                                for l in range(2):
                                                    if json.dumps([a,b,c,d,e,f,g,h,i,j,k,l]) not in final_encodings:
                                                        inverse_encodings.add(json.dumps([a,b,c,d,e,f,g,h,i,j,k,l]))
    print(len(inverse_encodings), len(final_encodings))
    print(len(memes),memes)
    return final_encodings, inverse_encodings
