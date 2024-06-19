#!/usr/bin/python3

import sys, os, shutil, re
from pathlib import Path

shutil.copyfile(f'docs/NEXT_README.md', f'README.md')
os.remove(f'docs/NEXT_README.md')
os.remove(f'docs/README.md')

if (len(sys.argv) != 2):
    raise Exception('usage: python3 bootstrap-template.py <name_for_extension_in_snake_case>')

name_extension = sys.argv[1]

def is_snake_case(s):
    # Define the regex pattern for snake case with numbers
    pattern = r'^[a-z0-9]+(_[a-z0-9]+)*$'

    # Use re.match to check if the string matches the pattern
    if re.match(pattern, s):
        return True
    else:
        return False

if name_extension[0].isdigit():
    raise Exception('Please dont start your extension name with a number.')

if not is_snake_case(name_extension):
    raise Exception('Please enter the name of your extension in valid snake_case containing only lower case letters and numbers')

def to_camel_case(snake_str):
    return "".join(x.capitalize() for x in snake_str.lower().split("_"))

def replace(file_name, to_find, to_replace):
    with open(file_name, 'r', encoding="utf8") as file :
        filedata = file.read()
    filedata = filedata.replace(to_find, to_replace)
    with open(file_name, 'w', encoding="utf8") as file:
        file.write(filedata)

files_to_search = []
files_to_search.extend(Path('./.github').rglob('./**/*.yml'))
files_to_search.extend(Path('./test').rglob('./**/*.test'))
files_to_search.extend(Path('./src').rglob('./**/*.hpp'))
files_to_search.extend(Path('./src').rglob('./**/*.cpp'))
files_to_search.extend(Path('./src').rglob('./**/*.txt'))
files_to_search.extend(Path('./src').rglob('./*.md'))

def replace_everywhere(to_find, to_replace):
    for path in files_to_search:
        replace(path, to_find, to_replace)
        replace(path, to_find.capitalize(), to_camel_case(to_replace))
        replace(path, to_find.upper(), to_replace.upper())
    
    replace("./CMakeLists.txt", to_find, to_replace)
    replace("./Makefile", to_find, to_replace)
    replace("./Makefile", to_find.capitalize(), to_camel_case(to_replace))
    replace("./Makefile", to_find.upper(), to_replace.upper())
    replace("./README.md", to_find, to_replace)
    replace("./extension_config.cmake", to_find, to_replace)

replace_everywhere("quack", name_extension)
replace_everywhere("Quack", name_extension.capitalize())
replace_everywhere("<extension_name>", name_extension)

string_to_replace = name_extension
string_to_find = "quack"

# rename files
os.rename(f'test/sql/{string_to_find}.test', f'test/sql/{string_to_replace}.test')
os.rename(f'src/{string_to_find}_extension.cpp', f'src/{string_to_replace}_extension.cpp')
os.rename(f'src/include/{string_to_find}_extension.hpp', f'src/include/{string_to_replace}_extension.hpp')

# remove template-specific files
os.remove('.github/workflows/ExtensionTemplate.yml')

# finally, remove this bootstrap file
os.remove(__file__)