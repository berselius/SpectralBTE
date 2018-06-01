# SpectralBTE

This code was originally developed at the University of Texas at Austin by Jeff Haack, and implements the 0D-3V and 1D-3V spectral Boltzmann code described in the following papers:

* I.M. Gamba and Sri Harsha Tharkabhushaman, 'Spectral-Lagrangian based methods applied to computation of non-equilibrium statistical states,' J. Comput. Phys., 228, 2012--2036, 2009.
* I.M. Gamba and Sri Harsha Tharkabhushaman, 'Shock and Boundary Structure formation by Spectral-Lagrangian methods for the Inhomogeneous Boltzmann Transport Equation', J. Comp. Math, 28, pp. 430--460, 2010.
* I.M. Gamba and J.R.Haack, 'A conservative spectral method for the Boltzmann equation with anisotropic scattering and the grazing collisions limit', J. Comput. Phys., 270, 40--57 (2014).

## Dependencies

The following C Libraries required to build this code 

* fftw3
* GSL (Gnu Scientific Library)
* OpenMP (should be standard in all gcc these days)
* MPI of some flavor


## Building the code

Before building the code, create a `obj` directory 

Simply type `make` and it will compile the code and create the program `boltz_`

To build an experimental faster (but less accurate) version of the code, type `make fast`. This will create the executable `Fastboltz_`

To build a standalone program that precomputes weights for anisotropic cross sections (and the Landau collision operator), type `make weights`. This will create the executable `WeightGen_`

## Running the code

The `boltz_` executable expects the following directories to be co-located with itself, you need to create them:

* input/
* Weights/
* Data/

The code looks for input files in the input directory

Output is stored in the Data directory

The Weights directory is where the code will look to check to see if you have already precomputed the collision weights for your input setup. If not, it will compute them and store them here for future use.

### Input file setup

