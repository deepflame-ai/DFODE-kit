import re

def is_numeric_string(input_string):
    pattern = r'^[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?$'

    if re.match(pattern, input_string):
        return True
    else:
        return False
    
