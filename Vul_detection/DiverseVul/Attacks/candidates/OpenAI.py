import random
import string
import time
from openai import OpenAI


def retry_request(request_function, max_retries=10, delay=20):
    for i in range(max_retries):
        try:
            print('request_function i=', i)
            response = request_function()
            return response
        except Exception as e:
            print(f"Attempt {i + 1} failed: {e}")
            time.sleep(5)
    raise Exception("Max retries reached, request failed")


def load_deepseek_coder_process_code(prompt):
    client = OpenAI(api_key="sk-9557afd63b4644ad97c6f490e22fcaaa", base_url="https://api.deepseek.com/")

    try:
        response = client.chat.completions.create(
            model="deepseek-coder",
            # model="deepseek-chat",
            messages=[
                {"role": "system", "content": ""},
                {"role": "user", "content": prompt},
            ],
            stream=True,
            temperature=0.0,
            # top_p=0.1,
            timeout=30,
        )
        # print(response)

        result = []
        for chunk in response:
            content = chunk.choices[0].delta.content if hasattr(chunk.choices[0].delta, 'content') else ''
            # print(content)
            if content:
                result.append(content)

        return ''.join(result)

    except Exception as e:
        print('Exception', e)
        return None


def extract_between_brackets(s):
    start = s.find('[') + 1
    end = s.find(']', start)
    if start > 0 and end > start:
        return s[start:end]
    return None


def generate_dead_code():
    random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=12))

    candidate_k = 10

    prompt = f'Generate {str(candidate_k)} complex dead code snippets that can be injected into a C function, ' \
             f'each containing multiple operations or statements,' \
             f'such as nested if-else, while or for loops with inner operations, ' \
             f'try-catch blocks with embedded function calls. ' \
             f'Ensure that all variable names are declared, consist of multiple characters, ' \
             f'and are random strings composed of both numbers and letters. ' \
             f'Each snippet should be one line long and include multiple statements and variable names. ' \
             f'Return only a json list without any explanation.{random_string}'

    for i in range(500):

        # candidate_site = load_deepseek_coder(prompt)
        candidate_site = load_deepseek_coder_process_code(prompt)

        print(candidate_site)

        if '```json' in candidate_site or 'json' in candidate_site:

            first_bracket_index = candidate_site.find('[')
            last_bracket_index = candidate_site.rfind(']')

            content_within_brackets = candidate_site[first_bracket_index + 1:last_bracket_index]
            quoted_content = content_within_brackets.strip().split('}",')

            candidate_site_ls_temp = []

            for id, item in enumerate(quoted_content):

                if id != len(quoted_content) - 1:
                    temp = item.strip() + '}'
                else:
                    temp = item.strip()
                candidate_site_ls_temp.append(temp)

            candidate_site_ls = []
            for id, code in enumerate(candidate_site_ls_temp):
                dead_code = code.replace('Dead', 'Useful').replace('never ', '').replace('dead',
                                                                                         'useful').replace(
                    'unused', 'used').replace('\\', '')
                if id == len(candidate_site_ls_temp) - 1:
                    dead_code = dead_code[1:-1]
                else:
                    dead_code = dead_code[1:]

                candidate_site_ls.append(dead_code)

            sorted_strings = sorted(candidate_site_ls, key=len)
            candidate_site_ls = sorted_strings

            if len(candidate_site_ls) != candidate_k:
                result = []
                for item in candidate_site_ls:
                    if item[0:3] == 'do ':
                        temp_item = item.split('",\n    "')
                        if len(temp_item) == 2:
                            result.append(temp_item[0])
                            result.append(temp_item[1])
                        else:
                            result.append(temp_item[0])
                    else:
                        result.append(item)

                candidate_site_ls = result

            unique_list = []

            for line in candidate_site_ls:

                result = r' \n '
                for id, char in enumerate(line):
                    if char in [';', '{', '}']:
                        result += char + r' \n '
                    else:
                        result += char

                print(result)

                unique_list.append(result)

            file_path = './dead_code_hub_large.txt'

            with open(file_path, 'a+', encoding='utf-8') as file:
                for element in unique_list:
                    file.write(element + "\n")


if __name__ == '__main__':
    generate_dead_code()
