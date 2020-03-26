# Setup tools for Lyncs

In this package we provide various setup tools used by Lyncs.

## Installation

The package can be installed via `pip`:

```
pip install --user lyncs_setuptools
```

## Usage

Lyncs' setuptools try to find most of the common setup parameters and defines its own `setup` function as in setuptools.

One can use in a `setup.py` file the following script

```
from lyncs_setuptools import setup

setup(package_name, **kwargs)
```

where package_name is the name of the package and kwargs are a list of arguments replacing or adding to the one automatically deduced by lyncs_setuptools.

For seeing the list of automatically deduced otpions just run `lyncs_setuptools` from command line.

[lyncs_setuptools]: from lyncs_setuptools import main; print("```\n>>> lyncs_setuptools \n\n"+main()+"\n'''")