#!/usr/bin/env python
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy
import os
from itertools import groupby
from distutils.core import setup
from distutils.extension import Extension

import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

includes = [ numpy.get_include() ]

extArgs = {
    "extra_compile_args" : ['-fopenmp'],
    "extra_link_args": ['-fopenmp'],
    "language" : "c++",
    
    "include_dirs" : includes,
    
    "libraries" : ["m"]
}

def find_pyx():
    ext_files = []
    for root,dirs,files in os.walk("seamr"):
        for x in files:
            if x.endswith(".pyx"):
                ext_files.append( os.path.join(root, x) )
    
    for d, pth in groupby(ext_files, key=lambda p: p.split(".")[0]):
        yield d, list(pth)

def extension(tup):
    print(tup[0].replace("/","."), tup[1])
    return Extension(tup[0].replace("/","."),
                     tup[1],
                     **extArgs)

ext_modules = list(map(extension, find_pyx()))

setup(name='seamr',
      version='0.5.0',
      description='SEmantic Activity Modeling and Recognition',
      author='Zachary Wemlinger',
      author_email='z.wemlinger@gmail.com',
      packages=['seamr', 'seamr.cli','seamr.features','seamr.learn', 'seamr.dl'],
      package_data={'seamr': ['ontology/*.ttl', 'graphs/*.json']},
      ext_modules = ext_modules,
      cmdclass = {'build_ext': build_ext},
      install_requires = ["cython",
                          "coloredlogs",
                          "tabulate",
                          "rdflib",
                          "pyyaml",
                          "numpy",
                          "scipy",
                          "tqdm",
                          "pyprg",
                          "cymem",
                          "python-crfsuite",
                          "seqlearn",
                          "hmmlearn",
                          "networkx",
                          "sklearn",
                          "scikit-image",
                          "scikit-video",
                          "pandas"
                         ]
     )
