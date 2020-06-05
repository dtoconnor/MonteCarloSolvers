from setuptools import setup, find_packages
# from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

extensions = [
    Extension(
        "piqmc.sa", ["piqmc/sa.pyx"],
        include_dirs=[numpy.get_include()],
        # extra_compile_args=['-fopenmp'],
        # extra_link_args=['-fopenmp'],
        language='c',
        ),
    Extension(
        "piqmc.qmc", ["piqmc/qmc.pyx"],
        include_dirs=[numpy.get_include()],
        # extra_compile_args=['-fopenmp'],
        # extra_link_args=['-fopenmp'],
        language='c',
        ),
    Extension(
        "piqmc.tools", ["piqmc/tools.pyx"],
        include_dirs=[numpy.get_include()],
        language='c',
        )
    ]

setup(
    name="piqmc",
    description="Path-integral quantum Monte Carlo code for simulating quantum annealing.",
    author="Daniel O'Connor / Hadayat Seddiqi ",
    author_email="uceedto@ucl.ac.uk / hadsed@gmail.com",
    url="https://github.com/dtoconnor/pathintegral-qmc",
    packages=find_packages(exclude=['testing', 'examples']),
    cmdclass={'build_ext': build_ext},
    ext_modules=extensions,
)
# for windows build you require Visual studio 15 or greater (tested on VS 17)
# make sure you have desktop development packages such that you have the
# cd to the directory with the setup.py then run the command below in the x86 native tools command prompt for VS 17
# python.exe setup.py build_ext --inplace --compiler=msvc


