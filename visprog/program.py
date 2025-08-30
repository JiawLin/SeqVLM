from visprog.utils import parse_step

import visprog.step_interpreters
import visprog.view_interpreters
import visprog.loc_interpreter
from visprog.registry import interpreter_registry


class Program:
    
    def __init__(self, prog_str, init_state=None):
        self.prog_str = prog_str.rstrip()
        self.state = init_state if init_state is not None else dict()
        self.instructions = self.prog_str.split('\n')


class ProgramInterpreter:
    
    def __init__(self, loc='LOC_BLIP'):
        self.interpreters = interpreter_registry
        self.interpreters['LOC'] = self.interpreters[loc]
        
        
    def execute_step(self, prog_step):
        step_name = parse_step(prog_step.prog_str, partial=True)['step_name']
        return self.interpreters[step_name].execute(prog_step)


    def execute(self, prog, init_state):
        if isinstance(prog, str):
            prog = Program(prog, init_state)
        else:
            assert (isinstance(prog, Program))

        prog_steps = [Program(instruction, init_state=prog.state) for instruction in prog.instructions]

        for prog_step in prog_steps:
            step_output = self.execute_step(prog_step)

        # print(prog.state)
        return step_output
