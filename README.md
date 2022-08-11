# PINION
***A featherweight EoR simulation***

# Goal
This project aims at accelerating radiative transfer simulation with physics informed neural network. For this, a CNN with an additional ODE loss is trained on a $\mbox{C}^2\mbox{-Ray}$ simulation. The end goal is to be able to produce highly detailed large scale simulations at lower computational cost. 

## Repository structure
At the root, one can find several Python script that contain definition of methods and classes that are used in the training. In particular `central_cnn.py` contains the definition of the neural network used in this project.

In the `trainings/` folder, one can find three jupyter notebooks. Two of them are used to train the neural network, and the last one allows to generate the propagating mask used as input from the network.

In the `prediction/` folder, one can find Python script to export subvolumes and group them at the end, as well as a typical slurm script to run the whole prediction.

## Installation
This project uses Python3 and the packages are listed in `requirements.txt`. This project was tested a GPU-equipped linux device running python 3.9.10. 

## What does PINION stands for ?
PINION is the acronym for **P**hysics **I**nformed **N**eural network for re**ION**ization.
