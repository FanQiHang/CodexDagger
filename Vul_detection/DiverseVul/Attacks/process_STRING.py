import re


def replace_between_hashes(s, flag, replacement):
    start = s.find(flag)
    if start != -1:
        end = s.find(flag, start + len(flag))
        if end != -1:
            return s[:start] + replacement + s[end + len(flag):]
    return s


def remove_comments(text):
    pattern = re.compile(r'/\*.*?\*/|/\*.*', re.DOTALL)
    return pattern.sub('', text)


def find_nearest_newlines(lst, index):
    before_newline = -1
    after_newline = -1

    for i in range(index - 1, -1, -1):
        if lst[i] == '\n':
            before_newline = i
            break

    for i in range(index + 1, len(lst)):
        if lst[i] == '\n':
            after_newline = i
            break

    return before_newline, after_newline


cpp_keywords = [
    "alignas", "alignof", "and", "and_eq", "asm", "atomic_cancel", "atomic_commit",
    "atomic_noexcept", "auto", "bitand", "bitor", "bool", "break", "case", "catch",
    "char", "char8_t", "char16_t", "char32_t", "class", "compl", "concept", "const",
    "consteval", "constexpr", "constinit", "const_cast", "continue", "co_await",
    "co_return", "co_yield", "decltype", "default", "delete", "do", "double",
    "dynamic_cast", "else", "enum", "explicit", "export", "extern",
    "float", "friend", "goto", "inline", "int", "long", "mutable",
    "namespace", "new", "noexcept", "not", "not_eq", "nullptr", "operator",
    "or", "or_eq", "private", "protected", "public", "reflexpr", "register",
    "reinterpret_cast", "requires", "return", "short", "signed", "sizeof",
    "static", "static_assert", "static_cast", "struct", "switch", "synchronized",
    "template", "this", "thread_local", "throw", "try", "typedef",
    "typeid", "typename", "union", "unsigned", "using", "virtual", "void",
    "volatile", "wchar_t", "xor", "xor_eq"
]

cpp_characters = [
    '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/',
    ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '{', '|', '}', '~',
    '\a', '\b', '\f', '\n', '\r', '\t', '\v', '\\', '\'', '\"', '\?', '\0', ' '
]


def contains_cpp_characters(s):
    for char in cpp_characters:
        if char in s:
            return True
    return False


def is_java_keyword(word):
    pattern = re.compile(r'\s*(' + '|'.join(cpp_keywords) + r')\s*')
    # Check if the word matches the pattern
    if pattern.fullmatch(word):
        return True
    return False


def find_last_right_brace_index(input_list):
    for index in range(len(input_list) - 1, -1, -1):
        if input_list[index] == '}':
            return index
    return -1


def is_valid_c_variable(word):
    pattern = r'^[a-zA-Z_][a-zA-Z0-9_]*$'
    if re.match(pattern, word):
        if word == "_":
            return False
        return True
    return False


def is_previous_word_else(code_list, if_index):
    if if_index <= 0:
        return False
    for i in range(if_index - 1, -1, -1):
        if code_list[i].strip() != '':
            return code_list[i].strip() == 'else'

    return False


def not_contains_letter_or_digit(s):
    flag = bool(re.search(r'[a-zA-Z0-9]', s))
    if flag:
        return False
    else:
        return True


def is_valid_c_integer(expression):
    pattern = r'^[+-]?(\d+|0[xX][0-9a-fA-F]+|0[0-7]+)$'
    return bool(re.match(pattern, expression))


def is_valid_c_float(expression):
    pattern = r'^[+-]?(\d*\.\d+|\d+\.\d*)([eE][+-]?\d+)?[fF]?$'
    return bool(re.match(pattern, expression))


def find_all_substring_bounds(main_string, substring):
    start_index = 0
    indices = []
    while True:
        start_index = main_string.find(substring, start_index)
        if start_index == -1:
            break
        end_index = start_index + len(substring) - 1
        indices.append((start_index, end_index))
        start_index += 1
    return indices
