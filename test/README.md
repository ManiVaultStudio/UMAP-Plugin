# UMAP Plugin Test
Run `python create_reference_data.py` in this directory to print reference values used in the tests.

The reference values are created in python using [umap-learn](https://umap-learn.readthedocs.io) and [numpy](https://numpy.org/).

## Dependencies:
The test setup depends on the plugin project. 
One extra dependencies is automatically downloaded during the cmake configuration: 
- [Catch2](https://github.com/catchorg/Catch2) for unit testing
