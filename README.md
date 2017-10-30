# larcv-tutorial
Tutorial of LArCV data products and interface with tensorflow.

This repository is meant to collect and share some basic steps for using larcv and training deep neural networks on lartpc data.  There are examples of Jupyter notebooks to see how the data interface works, simple tensorflow tutorials to connect larcv interface to tensorflow, and some small example files to get started and be able to run the tutorials easily.

# Requirements:
To use this repository, the main requirement is root (with PyRoot).  You may choose to download and install root from the origin (https://root.cern.ch/), or you may have it from an experiment software distribution.  You'll also need larcv2 (Available from the DeepLearnPhysics group), and for some of the tutorials you'll need jupyter, and tutorials that highlight deep learning require tensorflow.  If you have a GPU available you can do network training (which requires more example files), but without a GPU you can still interface with the data and build network graphs.
