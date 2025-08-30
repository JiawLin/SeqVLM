import logging
import io, tokenize
import numpy as np


def parse_step(step_str, partial=False):
    tokens = list(tokenize.generate_tokens(io.StringIO(step_str).readline))
    output_var = tokens[0].string
    step_name = tokens[2].string
    parsed_result = dict(output_var=output_var, step_name=step_name)
    if partial:
        return parsed_result

    arg_tokens = tokens[4:-3]
    args = dict()
    i = 0
    while i < len(arg_tokens):
        key = arg_tokens[i].string
        if arg_tokens[i + 1].string == '=':
            value_token = arg_tokens[i + 2]
            if value_token.string == '[':     # Handle list arguments
                list_values = []
                i += 3     # Move to the next token after the '=' and '['
                while arg_tokens[i].string != ']':
                    if arg_tokens[i].string != ',':
                        list_values.append(arg_tokens[i].string)
                    i += 1     # Move to the next token within the list
                value = list_values
                i += 1     # Move past the closing ']'
            else:
                value = value_token.string
                i += 2     # Move to the next key-value pair
        args[key] = value
        i += 1     # Move to the next key in the key-value pair

        # Skip the comma token outside of lists
        if i < len(arg_tokens) and arg_tokens[i].string == ',':
            i += 1

    parsed_result['args'] = args
    return parsed_result