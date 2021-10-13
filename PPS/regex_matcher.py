"""
cd PPS
python regex_matcher.py

For debuggers :
python regex_matcher.py > log.txt
"""

import os, fnmatch

import json

def find(pattern, path):
    """Utility to find files wrt a regex search"""
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result

def getBaseNameNoExt(givenPath):
    """Returns the basename of the file without the extension"""
    filename = os.path.splitext(os.path.basename(givenPath))[0]
    return filename

def read_notebook(file, store_markdown= False):
    """Reads a notebook file and returns the code"""

    code = json.load(open(file))
    # file = getBaseNameNoExt(file)
    py_file = "" # open(f"{file}.py", "w+")

    for cell in code['cells']:
        if cell['cell_type'] == 'code':
            for line in cell['source']:
                py_file += line # py_file.write(line)
            py_file += "\n" # py_file.write("\n")
        elif cell['cell_type'] == 'markdown' and store_markdown:
            py_file += "\n" # py_file.write("\n")
        
            for line in cell['source']:
                if line and line[0] == "#":
                    py_file += line # py_file.write(line)
            py_file += "\n" # py_file.write("\n")
    # py_file.close()
    
    return py_file

def read_python(file):
    """Reads a python file and returns the code"""
    py_file = open(file, "r")
    code = py_file.read()
    py_file.close()
    return code

def extract_files(FIND_FOLDER, debug=False, v = False):
    if v:
        print("Searching for files in :", FIND_FOLDER)
    py_files=find('*.py', FIND_FOLDER)
    ipynb_files = find('*.ipynb', FIND_FOLDER)
    if v:
        print("-"*15 + f'Found : {len(py_files)} python files | {len(ipynb_files)} ipynb files' + "-"*15)

    FOLDER_NAME = FIND_FOLDER.split(os.path.sep)[-1]
    UP_FOLDER = os.path.dirname(FIND_FOLDER)
    if debug:
        TempPPS = os.path.join(UP_FOLDER, ".$" + "TempPPS" + FOLDER_NAME)
        os.makedirs(TempPPS, exist_ok=True)
    # print(py_files)
    ALL_LINES = ""
    for file in py_files:
        if v:
            print(f"Reading : {file}")
        py_file = read_python(file)
        ALL_LINES+= py_file
    for file in ipynb_files:
        if v:
            print(f"Reading : {file}")
        py_file = read_notebook(file)
        ALL_LINES+= py_file            
        if debug:
            py_file_name = getBaseNameNoExt(file)
            py_file_path = os.path.join(TempPPS,f"{py_file_name}.py") # f"{FIND_FOLDER}/{py_file_name}.py"
            if v:
                print(f"Writing : {py_file_path}")
            with open(py_file_path, "w+") as f:
                f.write(py_file)
    if debug:
        all_file_path = os.path.join(TempPPS,f"ALL_FILES_CODE_DUMP.py") # f"{FIND_FOLDER}/{py_file_name}.py"
        if v:
            print(f"Writing : {all_file_path}")
        with open(all_file_path, "w+") as f:
            f.write(ALL_LINES)
    return ALL_LINES

if __name__ == '__main__':
    CURRENT_FOLDER = os.path.dirname(os.path.abspath(__file__)) # os.getcwd()
    UP_FOLDER = os.path.dirname(CURRENT_FOLDER)
    FIND_FOLDER = os.path.join(UP_FOLDER, "SampleFiles")
    all_code = extract_files(FIND_FOLDER, v=1, debug=1)
    



