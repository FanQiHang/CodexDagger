import random
import string


def generate_var():
    first = random.choice(string.ascii_letters + '_')
    rest = ''.join(random.choices(string.ascii_letters + string.digits + '_', k=random.randint(2, 5)))
    return first + rest


with open("redundant_code_augment.txt", "w") as f:
    used_vars = set()
    for _ in range(500):
        x = generate_var()
        while x in used_vars:
            x = generate_var()
        used_vars.add(x)

        y = generate_var()
        while y == x or y in used_vars:
            y = generate_var()
        used_vars.add(y)

        f.write(f"\\n int {y} =1; if (0) {{ \\n {y} +=1; }}\\n \n")
