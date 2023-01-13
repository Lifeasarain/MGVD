import re
import codecs
import os
import pandas as pd


# Clean Gadget
# Author https://github.com/johnb110/VDPython:
# For each gadget, replaces all user variables with "VAR#" and user functions with "FUN#"
# Removes content from string and character literals keywords up to C11 and C++17; immutable set
from typing import List

keywords = frozenset({'__asm', '__builtin', '__cdecl', '__declspec', '__except', '__export', '__far16', '__far32',
                      '__fastcall', '__finally', '__import', '__inline', '__int16', '__int32', '__int64', '__int8',
                      '__leave', '__optlink', '__packed', '__pascal', '__stdcall', '__system', '__thread', '__try',
                      '__unaligned', '_asm', '_Builtin', '_Cdecl', '_declspec', '_except', '_Export', '_Far16',
                      '_Far32', '_Fastcall', '_finally', '_Import', '_inline', '_int16', '_int32', '_int64',
                      '_int8', '_leave', '_Optlink', '_Packed', '_Pascal', '_stdcall', '_System', '_try', 'alignas',
                      'alignof', 'and', 'and_eq', 'asm', 'auto', 'bitand', 'bitor', 'bool', 'break', 'case',
                      'catch', 'char', 'char16_t', 'char32_t', 'class', 'compl', 'const', 'const_cast', 'constexpr',
                      'continue', 'decltype', 'default', 'delete', 'do', 'double', 'dynamic_cast', 'else', 'enum',
                      'explicit', 'export', 'extern', 'false', 'final', 'float', 'for', 'friend', 'goto', 'if',
                      'inline', 'int', 'long', 'mutable', 'namespace', 'new', 'noexcept', 'not', 'not_eq', 'nullptr',
                      'operator', 'or', 'or_eq', 'override', 'private', 'protected', 'public', 'register',
                      'reinterpret_cast', 'return', 'short', 'signed', 'sizeof', 'static', 'static_assert',
                      'static_cast', 'struct', 'switch', 'template', 'this', 'thread_local', 'throw', 'true', 'try',
                      'typedef', 'typeid', 'typename', 'union', 'unsigned', 'using', 'virtual', 'void', 'volatile',
                      'wchar_t', 'while', 'xor', 'xor_eq', 'NULL', 'printf', 'STR'})
# holds known non-user-defined functions; immutable set
main_set = frozenset({'main'})
# arguments in main function; immutable set
main_args = frozenset({'argc', 'argv'})

operators3 = {'<<=', '>>='}
operators2 = {
    '->', '++', '--', '**',
    '!~', '<<', '>>', '<=', '>=',
    '==', '!=', '&&', '||', '+=',
    '-=', '*=', '/=', '%=', '&=', '^=', '|='
}
operators1 = {
    '(', ')', '[', ']', '.',
    '+', '&',
    '%', '<', '>', '^', '|',
    '=', ',', '?', ':',
    '{', '}', '!', '~'
}


def to_regex(lst):
    return r'|'.join([f"({re.escape(el)})" for el in lst])


regex_split_operators = to_regex(operators3) + to_regex(operators2) + to_regex(operators1)

def _removeComments(source) -> []:
    in_block = False
    new_source = []
    # source = source.split('\n')
    for line in source:
        i = 0
        if not in_block:
            newline = []
        while i < len(line):
            if line[i:i + 2] == '/*' and not in_block:
                in_block = True
                i += 1
            elif line[i:i + 2] == '*/' and in_block:
                in_block = False
                i += 1
            elif not in_block and line[i:i + 2] == '//':
                break
            elif not in_block:
                newline.append(line[i])
            i += 1
        if newline and not in_block:
            new_source.append("".join(newline))
    return new_source


# input is a list of string lines
def clean_gadget(gadget):
    # dictionary; map function name to symbol name + number
    fun_symbols = {}
    # dictionary; map variable name to symbol name + number
    var_symbols = {}

    fun_count = 1
    var_count = 1

    # regular expression to find function name candidates
    rx_fun = re.compile(r'\b([_A-Za-z]\w*)\b(?=\s*\()')
    # regular expression to find variable name candidates
    # rx_var = re.compile(r'\b([_A-Za-z]\w*)\b(?!\s*\()')
    rx_var = re.compile(r'\b([_A-Za-z]\w*)\b((?!\s*\**\w+))(?!\s*\()')

    # final cleaned gadget output to return to interface
    cleaned_gadget = []

    for line in gadget:
        # replace any non-ASCII characters with empty string
        ascii_line = re.sub(r'[^\x00-\x7f]', r'', line)
        # remove all hexadecimal literals
        hex_line = re.sub(r'0[xX][0-9a-fA-F]+', "HEX", ascii_line)
        # return, in order, all regex matches at string list; preserves order for semantics
        user_fun = rx_fun.findall(hex_line)
        user_var = rx_var.findall(hex_line)

        # Could easily make a "clean gadget" type class to prevent duplicate functionality
        # of creating/comparing symbol names for functions and variables in much the same way.
        # The comparison frozenset, symbol dictionaries, and counters would be class scope.
        # So would only need to pass a string list and a string literal for symbol names to
        # another function.
        for fun_name in user_fun:
            if len({fun_name}.difference(main_set)) != 0 and len({fun_name}.difference(keywords)) != 0:
                # check to see if function name already in dictionary
                if fun_name not in fun_symbols.keys():
                    fun_symbols[fun_name] = 'FUN' + str(fun_count)
                    fun_count += 1
                # ensure that only function name gets replaced (no variable name with same
                # identifier); uses positive lookforward
                hex_line = re.sub(r'\b(' + fun_name + r')\b(?=\s*\()', fun_symbols[fun_name], hex_line)

        for var_name in user_var:
            # next line is the nuanced difference between fun_name and var_name
            if len({var_name[0]}.difference(keywords)) != 0 and len({var_name[0]}.difference(main_args)) != 0:
                # check to see if variable name already in dictionary
                if var_name[0] not in var_symbols.keys():
                    var_symbols[var_name[0]] = 'VAR' + str(var_count)
                    var_count += 1
                # ensure that only variable name gets replaced (no function name with same
                # identifier); uses negative lookforward
                # print(var_name, gadget, user_var)
                hex_line = re.sub(r'\b(' + var_name[0] + r')\b(?:(?=\s*\w+\()|(?!\s*\w+))(?!\s*\()',
                                  var_symbols[var_name[0]], hex_line)

        cleaned_gadget.append(hex_line)
    # return the list of cleaned lines
    return cleaned_gadget


def normalize_code(data_path, store_path):
    files = os.listdir(data_path)
    files_num = len(files)
    count = 0
    if not os.path.exists(store_path):
        os.mkdir(store_path)
    for file in files:
        count = count + 1
        print("\r", end="")
        print("Process progress: {}%: ".format(count / files_num * 100), end="")
        path = data_path + '/' + file
        with open(path, "r") as f1:
            code = f1.read()
            gadget: List[str] = []
            # remove all string literals
            no_str_lit_line = re.sub(r'["]([^"\\\n]|\\.|\\\n)*["]', '"STR"', code)
            # remove all character literals
            no_char_lit_line = re.sub(r"'.*?'", "", no_str_lit_line)
            code = no_char_lit_line

            for line in code.splitlines():
                if line == '':
                    continue
                stripped = line.strip()
                # if "\\n\\n" in stripped: print(stripped)
                gadget.append(stripped)
            clean = _removeComments(gadget)
            clean = clean_gadget(clean)

            with open(store_path + "/" + file, 'w', encoding='utf-8') as f2:
                f2.writelines([line + '\n' for line in clean])


def normalize_code_csv(data_path, store_path):
    data = pd.read_csv(data_path)
    files_num = data.shape[0]
    count = 0

    normalize_code = []
    rc_raw_code = []
    for index, row in data.iterrows():
        count = count + 1
        print("\r", end="")
        print("Process progress: {}%: ".format(count / files_num * 100), end="")
        raw_code = row['code']
        code = row['code']
        gadget: List[str] = []
        # remove all string literals
        try:
            no_str_lit_line = re.sub(r'["]([^"\\\n]|\\.|\\\n)*["]', '"STR"', code)
        except:
            print(code)
        # remove all character literals
        no_char_lit_line = re.sub(r"'.*?'", "", no_str_lit_line)
        code = no_char_lit_line

        for line in code.splitlines():
            if line == '':
                continue
            stripped = line.strip()
            # if "\\n\\n" in stripped: print(stripped)
            gadget.append(stripped)
        clean = _removeComments(gadget)
        clean = clean_gadget(clean)

        normalize = ""
        for line in clean:
            normalize = normalize + line + '\n'
        normalize_code.append(normalize)

        raw_lines = []
        try:
            for line in raw_code.splitlines():
                if line == '':
                    continue
                stripped = line.strip()
                raw_lines.append(stripped)
        except:
            print(raw_code)
        rc_raw_lines = _removeComments(raw_lines)
        rc = ""
        for line in rc_raw_lines:
            rc = rc + line + '\n'
        rc_raw_code.append(rc)

    data["raw"] = rc_raw_code
    data["normalize"] = normalize_code
    data.to_csv(os.path.join(store_path, 'full_data.csv'))


def main():
    raw_csv = "/data/fcq_data/vul_study_project/dataset/fq/sgs/csv/raw_data.csv"
    store_path = "/data/fcq_data/vul_study_project/dataset/fq/sgs/raw/nc"
    normalize_code_csv(raw_csv, store_path)


if __name__ == "__main__":
    main()


