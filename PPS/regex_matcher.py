"""
cd PPS
python regex_matcher.py
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

if __name__ == '__main__':
    CURRENT_FOLDER = os.path.dirname(os.path.abspath(__file__)) # os.getcwd()
    UP_FOLDER = os.path.dirname(CURRENT_FOLDER)
    FIND_FOLDER = os.path.join(UP_FOLDER, "SampleFiles")
    print(FIND_FOLDER)
    py_files=find('*.py', FIND_FOLDER)
    ipynb_files = find('*.ipynb', FIND_FOLDER)
    
    print(f'Found {len(py_files)} python files')
    print(py_files)
    print(f'Found {len(ipynb_files)} ipynb files')

    for file in ipynb_files:
        py_file = read_notebook(file)
        py_file_name = getBaseNameNoExt(file)
        py_file_path = f"{py_file_name}.py" # f"{FIND_FOLDER}/{py_file_name}.py"
        print(f"Writing {py_file_path}")
        with open(py_file_path, "w+") as f:
            f.write(py_file)



