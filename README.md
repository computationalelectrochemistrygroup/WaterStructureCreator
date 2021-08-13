# watercreator patch for pymatgen

# Prerequisites (old)

- python==3.8.3 
- scipy==1.4.1
- numpy==1.17.5 
- matplotlib==3.1.3 
- ipython==7.12.0
- scikit-learn==0.24.1
- ase==3.20.1

# Installation

Clone or download the repository.

In the directory run:

`pip install -r requirements.txt`

This will install all necessary packages. You might need to comment out/delete the requirements for the python version, if you use conda, instead initialize the environment with `python==3.8.3`.


To install Pymatgen, download the version https://pypi.org/project/pymatgen/2020.11.11/. 

Selct the .tar.gz file, download it and put it into a suitable directory. 

In this directory run:

`tar zxvf pymatgen-2020.11.11.tar.gz `

`pip install -e pymatgen-2020.11.11`

To patch the program run:

`cd /path-to-pymatgen/pymatgen/analysis`

`ln -s  /path-to-gitrepo/waterstructureCreator .`
