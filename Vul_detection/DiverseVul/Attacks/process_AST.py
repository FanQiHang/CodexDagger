from parser import CPP_AST as AST
import re
from tree_sitter import Language, Parser
import tree_sitter_cpp as tscpp


def t_rename_func(the_code):
    """
    all_sites=True: a single, randomly selected, referenced field
    (self.field in Python) has its name replaced by a hole
    all_sites=False: all possible fields are selected
    """
    func_ls = []
    the_ast = AST.build_ast(the_code)
    func = the_ast.get_func_name()

    func_dict = AST.conv2dict(func)

    func_dict, code_list = AST.split_code(func_dict, the_code)

    for name in func_dict:
        func_ls.append(name)

    return func_ls


def t_rename_fields(the_code):
    """
    all_sites=True: a single, randomly selected, referenced field
    (self.field in Python) has its name replaced by a hole
    all_sites=False: all possible fields are selected
    """
    fields_ls = []
    the_ast = AST.build_ast(the_code)
    fields = the_ast.get_fields()

    fields_dict = AST.conv2dict(fields)
    fields_dict, code_list = AST.split_code(fields_dict, the_code)

    for name in fields_dict:
        fields_ls.append(name)

    return fields_ls


def t_rename_parameters(the_code):
    """
    Parameters get replaced by holes.
    """
    parameters_ls = []
    the_ast = AST.build_ast(the_code)
    parameters = the_ast.get_parameters()

    parameters = [par.source for par in parameters]
    parameters = the_ast.get_selected_parameters(parameters)

    parameters_dict = AST.conv2dict(parameters)
    parameters_dict, code_list = AST.split_code(parameters_dict, the_code)

    for name in parameters_dict:
        parameters_ls.append(name)

    return parameters_ls


def t_rename_local_variables(the_code):
    """
    Local variables get replaced by holes.
    """
    local_variables_ls = []
    the_ast = AST.build_ast(the_code)

    variables = the_ast.get_local_variables()

    variables = [par.source for par in variables]

    variables = the_ast.get_selected_variables(variables)

    variables_dict = AST.conv2dict(variables)

    variables_dict, code_list = AST.split_code(variables_dict, the_code)

    for name in variables_dict:
        local_variables_ls.append(name)

    return local_variables_ls


def t_replace_true_false(the_code):
    """
    Boolean literals are replaced by an equivalent
    expression containing a single hole
    (e.g., ("<HOLE>" == "<HOLE>") to replace true).
    """
    true_false_ls = []
    the_ast = AST.build_ast(the_code)
    booleans = the_ast.get_booleans()

    booleans_dict = AST.conv2dict(booleans)
    booleans_dict, code_list = AST.split_code(booleans_dict, the_code)

    for name in booleans_dict:
        true_false_ls.append(name)

    return true_false_ls


def t_insert_statements(the_code):
    """
    Statement of the form if False:\n <HOLE> = 1 is added to the target program.
    all_sites=False: The insertion location (either beginning, or end) is chosen at random.
    all_sites=True: The statement is inserted at all possible locations.
    """
    insert_statements_ls = []

    the_ast = AST.build_ast(the_code)

    statements = the_ast.get_statements()

    statements_dict = AST.conv2dict(statements)
    statements_dict, code_list = AST.split_code(statements_dict, the_code)

    for name in statements_dict:
        insert_statements_ls.append(name)

    return insert_statements_ls


def get_AST_set(the_code):
    # func_ls = t_rename_func(the_code)

    try:
        fields_ls = t_rename_fields(the_code)

        parameters_ls = t_rename_parameters(the_code)

        local_variables_ls = t_rename_local_variables(the_code)

        # true_false_ls = t_replace_true_false(the_code)

        # combined_ls = func_ls + fields_ls + parameters_ls + local_variables_ls + true_false_ls

        combined_ls = fields_ls + parameters_ls + local_variables_ls

        combined_set = set(combined_ls)
    except Exception as e:

        combined_set = set()

    return combined_set


def is_boolean_expression(java_code):

    boolean_pattern = re.compile(r'\b(true|false)\b')

    matches = boolean_pattern.findall(java_code)

    if matches:
        return True
    return False


def get_if_for_while_statements(key_value, code):

    LANGUAGE = Language(tscpp.language())
    parser = Parser(LANGUAGE)
    tree = parser.parse(bytes(code, 'utf8'))

    results = []

    def find_nested_if_statements(node, results):

        for child in node.children:
            if child.type == key_value + '_statement':
                results.append(code[child.start_byte: child.end_byte].strip())
                find_nested_if_statements(child, results)
            else:
                find_nested_if_statements(child, results)

    root_node = tree.root_node
    find_nested_if_statements(root_node, results)
    # print(key_value, results)
    return results


def get_statements_declaration(code):
    LANGUAGE = Language(tscpp.language())
    parser = Parser(LANGUAGE)
    tree = parser.parse(bytes(code, 'utf8'))

    results = []

    def find_nested_if_statements(node, results):
        if node.type in ['declaration']:
            results.append(code[node.start_byte: node.end_byte].strip())
        for child in node.children:
            find_nested_if_statements(child, results)

    root_node = tree.root_node
    find_nested_if_statements(root_node, results)
    # print('statements', results)
    return results


def get_statements(code):
    insert_statements_ls = []
    the_ast = AST.build_ast(code)
    statements = the_ast.get_statements()
    statements_dict = AST.conv2dict(statements)
    statements_dict, code_list = AST.split_code(statements_dict, code)
    # print(statements_dict)
    for name in statements_dict:
        insert_statements_ls.append(name)
    return insert_statements_ls