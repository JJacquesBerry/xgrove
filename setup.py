from setuptools import setup, find_packages
from pathlib import Path
import codecs
import os

VERSION = '0.2.3'
DESCRIPTION = 'creating and calculating groves of tree models'
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Setting up
setup(
    name="xgrove",
    version=VERSION,
    author="Jean Jacques Berry",
    author_email="<J.Jacques.Berry@gmail.com>",
    description=DESCRIPTION,long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=[
        "mkdocs",  # Für Dokumentation, falls erforderlich
        "scikit-learn>=0.24.1",
        "numpy>=1.19.2",
        "matplotlib>=3.3.2",
        "pandas>=1.1.3",
        "graphviz>=0.16",
        "statistics",  # Python eingebautes Modul, kein externes Package
        # Überprüfen, ob eventuell zusätzliche Module benötigt werden, die hier nicht explizit genannt wurden
    ],
    python_requires='>=3.6',
    keywords=['tree', 'model', 'grove', 'forest'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)