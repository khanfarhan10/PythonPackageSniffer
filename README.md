# Python Package Sniffer (PPS)
Package Sniffer for Python : Extract Package Names for Python Projects and Resolve Dependencies.

## Objective :

 
<blockquote>
 PPS is a Sniffer of Python Packages in a Directory. It used file structures to get you a complete list of installable packages for a fresh enviornment, with dependencies resolved.
</blockquote>

## Tasks to Solve :

- Have you ever worked on a project with a lot of scripts, notebooks and and a messy file structure? 
- Do you like to work with clean isolated virtual environments? 
- Do you have problems with dependency managements and frequently get import errors?

## Ideas :

- **Regex Solver** - Use `import x` from `*.py` and `*.ipynb` to check for package `x` in pip using cloud for the given directory. If found attach to file called `requirements.txt`.
- **Dependency Management & Resolver** - Use python poetry to get rid of program dependencies using latest python version preferably, or a fixed python version.
- **Smart Sense** - looks for various extensions from a smartfile to get the required package essentials. For example, big `h5` & `hdf5`, require package `h5py` for manipulating such files.
