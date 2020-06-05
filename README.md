# pathintegral-qmc
A path-integral quantum Monte Carlo code for simulating quantum annealing with arbitrary Ising Hamiltonians. It is written based on the 2002 Phys. Rev. B paper by Martonak, Santoro, and Tosatti entitled, 'Quantum annealing by the path-integral Monte Carlo method: The two-dimensional random Ising model' (you may find a free copy of this on arxiv.org).

There are also extensions to this paper by including System Bath coupling to capture dephasing effects as well as Wolff and Swendsen-Yang cluster updates.
## Requirements
This simulation package is written in Cython and requires ```scipy``` and ```numpy```. The C files are included with the .pyx. Installation requires ```setuptools```.

## Installation
### Linux
After cloning the repo, navigate to where you see ```setup.py``` and run ```python setup.py install```, or if you're developing (or wish to uninstall later) do ```python setup.py develop``` (where you can write ```python setup.py develop --uninstall``` if you wish to remove it later).
### Windows
After cloning the repo, check that you have Microsoft Visdual studio, and that you have the C builder extension installed such that your windows has the 'x86 Native Tools Command Prompt for VS XXXX'.
Open and move to the directory with ```setup.py``` and run ```python.exe setup.py build_ext --inplace --compiler=msvc```.

## Usage
See examples for usage
