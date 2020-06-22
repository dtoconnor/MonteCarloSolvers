from setuptools import setup, find_packages
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

extensions = [
    Extension(
        "solvers.sa", ["solvers/sa.pyx"],
        include_dirs=[numpy.get_include()],
        # extra_compile_args=['-fopenmp'],
        # extra_link_args=['-fopenmp'],
        language='c',
        ),
    Extension(
        "solvers.svmc", ["solvers/svmc.pyx"],
        include_dirs=[numpy.get_include()],
        # extra_compile_args=['-fopenmp'],
        # extra_link_args=['-fopenmp'],
        language='c',
        ),
    Extension(
        "solvers.qmc", ["solvers/qmc.pyx"],
        include_dirs=[numpy.get_include()],
        # extra_compile_args=['-fopenmp'],
        # extra_link_args=['-fopenmp'],
        language='c',
        ),
    Extension(
        "solvers.tools", ["solvers/tools.pyx"],
        include_dirs=[numpy.get_include()],
        language='c',
        )
    ]

setup(
    name="MCS",
    description="Set of Monte Carlo Solvers.",
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


