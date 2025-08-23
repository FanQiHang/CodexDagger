import re


def remove_comments_and_docstrings(source):
    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):
            return " "  # note: a space and not an empty string
        else:
            return s

    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    temp = []
    for x in re.sub(pattern, replacer, source).split('\n'):
        if x.strip() != "":
            temp.append(x)
    return '\n'.join(temp)


def is_empty_function(code):
    body_match = re.search(r'\{([\s\S]*?)\}', code)
    if not body_match:
        return False
    body = body_match.group(1).strip()

    body = re.sub(r'//.*|\/\*[\s\S]*?\*\/', '', body)
    body = body.strip()

    return not body


def is_single_line_function(code):
    body_match = re.search(r'\{([\s\S]*)\}', code)
    if not body_match:
        return False
    body = body_match.group(1).strip()

    body = re.sub(r'//.*|\/\*[\s\S]*?\*\/', '', body)
    body = body.strip()

    lines = [line.strip() for line in body.splitlines() if line.strip()]
    return len(lines) == 1
