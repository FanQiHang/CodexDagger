import json
import os
import random

import numpy as np
import torch

# import codeattack

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def html_style_from_dict(style_dict):
    """Turns.

        { 'color': 'red', 'height': '100px'}

    into
        style: "color: red; height: 100px"
    """
    style_str = ""
    for key in style_dict:
        style_str += key + ": " + style_dict[key] + ";"
    return 'style="{}"'.format(style_str)


def html_table_from_rows(rows, title=None, header=None, style_dict=None):
    # Stylize the container div.
    if style_dict:
        table_html = "<div {}>".format(html_style_from_dict(style_dict))
    else:
        table_html = "<div>"
    # Print the title string.
    if title:
        table_html += "<h1>{}</h1>".format(title)

    # Construct each row as HTML.
    table_html = '<table class="table">'
    if header:
        table_html += "<tr>"
        for element in header:
            table_html += "<th>"
            table_html += str(element)
            table_html += "</th>"
        table_html += "</tr>"
    for row in rows:
        table_html += "<tr>"
        for element in row:
            table_html += "<td>"
            table_html += str(element)
            table_html += "</td>"
        table_html += "</tr>"

    # Close the table and print to screen.
    table_html += "</table></div>"

    return table_html


def set_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)


def hashable(key):
    try:
        hash(key)
        return True
    except TypeError:
        return False


def sigmoid(n):
    return 1 / (1 + np.exp(-n))


GLOBAL_OBJECTS = {}
ARGS_SPLIT_TOKEN = "^"
