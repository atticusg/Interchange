import json

# copied from MultiplyQuantifiedData repo

class sentence:
    #this class stores the logical representation of a sentence and generates the information necessary
    #for both the first order logic model and the natural logic model
    def __init__(self, subject_noun, verb, object_noun, negate, adverb, subject_adjective, object_adjective, subject_determiner, object_determiner):
        self.subject_noun = subject_noun
        self.verb = verb
        self.object_noun = object_noun
        self.negate = negate
        self.subject_determiner = subject_determiner
        self.object_determiner = object_determiner
        self.string_object_determiner = object_determiner
        #if negate and object_determiner == "some":
        #    self.string_object_determiner = "any"
        self.negation = ""
        self.adverb = adverb
        self.subject_adjective = subject_adjective
        self.object_adjective = object_adjective
        self.verb_index = 0#ensures a non-negated verb is conjugated for first person present
        if negate:
            self.negation = "does not"
        self.string = self.construct_string([self.subject_determiner,self.subject_adjective,self.subject_noun,self.negation,self.adverb,self.verb[self.verb_index],self.string_object_determiner,self.object_adjective,self.object_noun])
        if self.negation == "":
            self.emptystring = self.construct_emptystring([self.subject_determiner,self.subject_adjective,self.subject_noun,self.negation,self.adverb,self.verb[self.verb_index],self.string_object_determiner,self.object_adjective,self.object_noun])
        else:
            self.emptystring = self.construct_emptystring([self.subject_determiner,self.subject_adjective,self.subject_noun,self.negation,self.adverb,self.verb[self.verb_index],self.string_object_determiner,self.object_adjective,self.object_noun])
        self.construct_logical_form_joint_predicates()
        self.initialize_natlog()

    def initialize_natlog(self):
        #This function decomposes "no" into negated "some" and "not every"
        #into negated every for the natural logic model
        if self.subject_determiner == "no":
            self.natlog_subject_determiner = "some"
            self.subject_negation = True
        if self.subject_determiner == "not every":
            self.natlog_subject_determiner = "every"
            self.subject_negation = True
        if self.subject_determiner == "some":
            self.natlog_subject_determiner = "some"
            self.subject_negation = False
        if self.subject_determiner == "every":
            self.natlog_subject_determiner = "every"
            self.subject_negation = False
        if self.object_determiner == "no":
            self.natlog_object_determiner = "some"
            self.object_negation = True
        if self.object_determiner == "not every":
            self.natlog_object_determiner = "every"
            self.object_negation = True
        if self.object_determiner == "some":
            self.natlog_object_determiner = "some"
            self.object_negation = False
        if self.object_determiner == "every":
            self.natlog_object_determiner = "every"
            self.object_negation = False
        if self.negation == "":
            self.verb_negation = False
        else:
            self.verb_negation = True

    def construct_string(self,lst):
        #turn a list of words into a single sentence string
        result = ""
        for word in lst:
            if word != "":
                result += word + " "
        return result[:-1]

    def construct_emptystring(self,lst):
        #turn a list of words into a single sentence string
        result = ""
        for word in lst:
            if word == "not every":
                result += "notevery" + " "
            elif word == "does not":
                result += "doesnot" + " "
            elif word != "":
                result += word + " "
            else:
                result += "emptystring" + " "
        return result[:-1]

    def construct_logical_form_joint_predicates(self):
        #construct a first order logic representation where adjectives and nouns are merged
        # into single predicates as well as adverbs and verbs
        logical_form = ""
        subject_noun_variable= "x"
        object_noun_variable= "y"
        verb_arg = "(" + subject_noun_variable + "," + object_noun_variable + ")"
        logical_form = self.verb[2] + verb_arg
        if self.adverb != "" :
            logical_form = self.adverb + logical_form
        object_logical_form = self.object_noun + "(" + object_noun_variable + ")"
        if self.object_adjective != "":
            object_logical_form = self.object_adjective + object_logical_form
        logical_form = self.add_quantifier(self.object_determiner, object_logical_form, logical_form, object_noun_variable)
        if self.negate:
            logical_form = "-" + "(" + logical_form + ")"
        subject_logical_form = self.subject_noun + "(" + subject_noun_variable + ")"
        if self.subject_adjective != "":
            subject_logical_form =self.subject_adjective + subject_logical_form
        self.logical_form = self.add_quantifier(self.subject_determiner, subject_logical_form, logical_form, subject_noun_variable)
        self.logical_form = "(" + self.logical_form + ")"
        self.assumptions= "exists x.(" + subject_logical_form + ") & exists y.(" + object_logical_form + ") & all y.(" + object_logical_form + "->" + self.object_noun + "(" + object_noun_variable + ")" + ")" + "& all x.(" + subject_logical_form + "->" + self.subject_noun + "(" + subject_noun_variable + ")" + ")" + "& all x.(all y.(" + self.adverb + self.verb[2] + verb_arg + "->" + self.verb[2] + verb_arg + "))"

    def construct_logical_form_single_predicates(self):
        #construct a first order logic representation where adjectives and nouns
        #are seperate predicates as well as adverbs and verbs
        logical_form = ""
        subject_noun_variable= "x"
        object_noun_variable= "y"
        verb_arg = "(" + subject_noun_variable + "," + object_noun_variable + ")"
        logical_form = self.verb[2] + verb_arg
        if self.adverb != "" :
            logical_form ="(" +logical_form +  "&" + self.adverb + verb_arg + ")"
        object_logical_form = self.object_noun + "(" + object_noun_variable + ")"
        if self.object_adjective != "":
            object_logical_form = "(" + object_logical_form + "&" + self.object_adjective + "(" + object_noun_variable + ")" + ")"
        logical_form = self.add_quantifier(self.object_determiner, object_logical_form, logical_form, object_noun_variable)
        if self.negate:
            logical_form = "-" + "(" + logical_form + ")"
        subject_logical_form = self.subject_noun + "(" + subject_noun_variable + ")"
        if self.subject_adjective != "":
            subject_logical_form ="(" + subject_logical_form +  "&" + self.subject_adjective + "(" + subject_noun_variable +  ")" + ")"
        self.logical_form = self.add_quantifier(self.subject_determiner, subject_logical_form, logical_form, subject_noun_variable)
        self.assumptions = "exists x.(" + subject_logical_form + ") & exists y.(" + object_logical_form + ")"

    def add_quantifier(self, determiner, first_expression, second_expression, variable):
        #takes in a determiner, two FOL expressions, and a variable and
        #returns a quantified FOL expression according to the arguments
        result = ""
        if determiner == "some" or determiner == "no":
            result = "exists " + variable + " .(" + first_expression+ "&" + second_expression+ ")"
            if determiner == "no":
                result = "-(" + result + ")"
        if determiner == "every" or determiner == "not every":
            result = "all " + variable + " .(" + first_expression+ "->" + second_expression+ ")"
            if determiner == "not every":
                result = "-(" + result + ")"
        return result



def parse_compound_sentence(data, input_sentence):
    #Takes a compound input_sentence and outputs the corresponding
    #instance of the sentence class for the first simple sentence
    #then the conjunction then the instance of the sentence class for
    #the second simple sentence
    conjunction = ""
    if " then " in input_sentence:
        return [parse_simple_sentence(data, input_sentence[3:input_sentence.index(" then ")])[0],"then", parse_simple_sentence(data,input_sentence[input_sentence.index(" then ")+6:])[0]]
    if " or " in input_sentence:
        return [parse_simple_sentence(data, input_sentence[:input_sentence.index(" or ")])[0],"or", parse_simple_sentence(data,input_sentence[input_sentence.index(" or ")+4:])[0]]
    if " and " in input_sentence:
        return [parse_simple_sentence(data, input_sentence[:input_sentence.index(" and ")])[0],"and", parse_simple_sentence(data,input_sentence[input_sentence.index(" and ")+5:])[0]]

def verify_parse(data, subject_noun, verb, object_noun, negation, adverb, subject_adjective, object_adjective, subject_determiner, object_determiner):
    if subject_noun not in data["agents"]:
        print("Subject noun is invalid")
        return False
    if object_noun not in data["things"]:
        print("Object noun is invalid")
        return False
    if verb not in data["transitive_verbs"]:
        print("Verb is invalid")
        return False
    if subject_adjective not in data["subject_adjectives"] + [""]:
        print("Subject adjective is invalid")
        return False
    if object_adjective not in data["object_adjectives"] + [""]:
        print("Object adjective is invalid")
        return False
    if adverb not in data["adverbs"] + [""]:
        print("Adverb is invalid")
        return False
    if subject_determiner not in ["not every", "no", "some", "every", "any"]:
        print("Subject determiner is invalid")
        return False
    if object_determiner not in ["not every", "no", "some", "every", "any"]:
        print("Object determiner is invalid")
        return False
    return True

def parse_simple_sentence(data, input_sentence):
    #Takes a simple input_sentence and outputs the corresponding
    #instance of the sentence class
    words = input_sentence.split()
    if words[0] == "notevery":
        subject_determiner = "not every"
        words = words[1:]
    else:
        subject_determiner = words[0]
        words = words[1:]
    if words[0] in data["subject_adjectives"]:
        subject_adjective = words[0]
        words = words[1:]
    else:
        subject_adjective = ""
        words = words[1:]
    subject_noun = words[0]
    words = words[1:]
    if words[0] == "doesnot":
        negation = True
        words = words[1:]
    else:
        negation = False
        words = words[1:]
    if words[0] in data["adverbs"]:
        adverb = words[0]
        words = words[1:]
    else:
        adverb = ""
        words = words[1:]
    verb = ""
    for verb_list in data["transitive_verbs"]:
        if words[0] in verb_list:
            verb = verb_list
    words = words[1:]
    if words[0] == "notevery":
        object_determiner = "not every"
        words = words[1:]
    else:
        object_determiner = words[0]
        words = words[1:]
    if words[0] in data["object_adjectives"]:
        object_adjective = words[0]
        words = words[1:]
    else:
        object_adjective = ""
        words = words[1:]
    object_noun = words[0]
    if not verify_parse(data, subject_noun, verb, object_noun, negation, adverb, subject_adjective, object_adjective, subject_determiner, object_determiner):
        return None
    return [sentence(subject_noun, verb, object_noun, negation, adverb, subject_adjective, object_adjective, subject_determiner, object_determiner)]

def parse_sentence(data, sentence):
    if " or " in sentence or " and " in sentence or " then " in sentence:
        return parse_compound_sentence(data, sentence)
    else:
        return parse_simple_sentence(data, sentence)
