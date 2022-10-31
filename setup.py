#!/usr/bin/env python

from setuptools import find_packages, setup

setup(name = 'graphnics',
      version = '1',
      description = 'Combining networkx and FEniCS to solve network models',
      author = 'Ingeborg G. Gjerde',
      author_email = 'ingeborg@simula.no',
      url = 'https://github.com/IngeborgGjerde/graphnics/',
      packages=find_packages(),
      include_package_data=True
)