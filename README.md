# SNCforSAM

This repository holds the code for the SAM memory model introduced in 
'KNN Classifier with Self Adjusting Memory for Heterogenous Concept Drift' from Losing et al.

An adaption of the SNC method from 'Stochastic Neighbor Compression' by Kusner et al. was added to the code.


# SAMkNN

## Installing nearest neighbor library:
Before using SAM you have to install a C++ library for the nearest neighbor calculations.
- Edit nearestNeighbor/setup.py and make sure that the variable "include_dirs" contains the path to your local python2.7 folder as well as the numpy folder.
- Install the C++ library by using the common setup.py script e.g. "python setup.py install --user".

## Using SAM
In SAMKNN/testSAMKNN.py you can find a test script which applies the SAMKNN model on a dataset. Simply execute "python testSAMKNN.py". 

## Datasets
Two exemplary datasets "weather" and "moving squares" can be found in the datasets folder. More drift datasets can be found in the repository https://github.com/vlosing/driftDatasets
