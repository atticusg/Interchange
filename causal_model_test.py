from causal_model import CausalModel

class AndTwiceProgram(CausalModel):
    def __init__(self):
        super().__init__([{True,False},{True,False},{True,False}],[{True,False}],[{True,False}])

    def run_model(self,input,intervention):
        input_tape, variable_tape, output_tape = self.initialize_tapes(input, intervention)
        variable_tape.set_value(0, input_tape.get_value(0) and input_tape.get_value(1),intervention)
        output_tape.set_value(0, variable_tape.get_value(0) and input_tape.get_value(2))
        variable_ordering = [0]
        return input_tape, variable_tape, output_tape, variable_ordering

def and_model_test():
    model = AndTwiceProgram()
    for input in [(a,b,c) for a in [True, False] for b in [True, False] for c in [True, False]]:
        for intervention in [{0:x} for x in [True,False]] + [dict()]:
            input_tape, variable_tape, output_tape = model.run_model(input, intervention)
            for i in range(3):
                if input[i] != input_tape.get_value(i):
                    print("input wrong")
            if 0 not in intervention:
                if input[0] and input[1] != variable_tape.get_value(0):
                    print("variable wrong")
                if input[0] and input[1] and input[2] != output_tape.get_value(0):
                    print("output no intervention wrong")
            else:
                if intervention[0] != variable_tape.get_value(0):
                    print("variable wrong")
                if intervention[0] and input[2] != output_tape.get_value(0):
                    print("output with intervention wrong")

if __name__ == '__main__':
    and_model_test()
