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

If running on Cori then load the following modules

* gsl
* cray-fftw
* openmpi

## Building the code

The code uses CMake. First create a build directory and change into it
`mkdir build`
`cd build`

Now run CMake
`cmake ..`

This assumes the build directory is immediately below the root direction,
otherwise run `cmake /path/to/root`.

Now build with
`make -j4`

And run tests with
`make test`

Note that the executables are located in the `exec` directory inside the build
directory.


## Running the code

### Some directory setup
The `boltz_` executable expects the following directories to be co-located with itself, you need to create them:

* `input/`
* `Weights/`
* `Data/`
* `restart/`
The code looks for input files in the input directory

Output is stored in the Data directory

The Weights directory is where the code will look to check to see if you have already precomputed the collision weights for your input setup. If not, it will compute them and store them here for future use.

If you tell the code to dump restart information, it will be stored in the restart folder

### To execute the code

Use `mpirun` to run the code, as

`mpirun -n <num_tasks> boltz_ input_file output_flags`

* The code will look for the specified `input_file` in `input/`. See below and in the examples firectory for details on writing input files.
* The code will look for the specified output flag directions in the specified `output_flags` in `input/`. This tells the code what variables you want to dump. See below for more details.
* If the restart flag is set in the input file, it will look for information in the `restart` folder.

#### How many MPI ranks, OMP threads should/can I use?

* If running a 0D/homogeneous problem, run with one MPI rank (`-n 1`).
* Set `OMP_NUM_THREADS` to the number of cores on one node
* For 1D/inhomogenous problems, the number of MPI ranks must evenly divide the number of spatial grid cells.
  * Past computational studies suggest that the most efficient way to set up the problem is to run with *one MPI rank per socket* and to set `OMP_NUM_THREADS` to be the number of cores per socket.

### Input file setup

The input file parser reads the file line by line looking for keywords that set up the problem

### keywords

* `N` - Number of velocity nodes / Fourier modes in each dimension. As this code is a 3V code, this means a total of N^3 velocity grid nodes / Fourier modes. (default: 16)
* `L_v` - Nondimensionalized width of velocity domain. (default: 5)
* `Knudsen` - The Knudsen number of the problem. This is the ratio of the mean free path to a characteristic macroscopic length scale. When the Knudsen number is small, one is close to a hydrodynamic limit. (default: 1)
* `Lambda` - exponent on the relative velocity in the cross section. For spheres-like (isotropic) kernels, 1 corresponds to hard spheres, 0 corresponds to Maxwell molecules, -3 corresponds to Coulomb like velocity scaling. (default: 1)
* `Time_step` - the non-dimensional time step. (default: -1, i.e. the code will exit if this is not set)
* `Number_of_time_steps` - How many steps to run until the code exits. (default: 1000)
* `Space_order` - 0 or 1. Seets the spatial discretization order for the LHS/Vlasov terms. 0 is an upwind scheme, 1 is a second order shock capturing method using a minmod slope limiter. (default: 1)
* `Data_writing_frequency` - the code will dump data at this rate (in timesteps) (default: 10)
* `Restart` - 0 or 1. If set to 1, this will restart the code from a previous dump. More info below. (default: 0)
* `Restart_time` - Amount of wall clock time (in seconds) until the code halts, generates restart info, then exits. This can then be restarted by running the code with the same input file but with a `Restart` value of 1. (default: 85500, i.e. 23 hours, 45 minutes). Set this to `-1` if you want the code to exit without creating a restart dump.
* `Init_field` - Flag for different initial data cases. See `src/initializer.c` for specifics...you may wish to change some of the parameters.
  * For 0D/space homogenous problems:
     * 0: Shifted isotropic problem
     * 1: Discontinous Maxwellian
     * 2: Bobylev-Krook-Wu problem
     * 3: Two Maxwellians
     * 4: Steady state test (initial condition is a Maxwellians)
     * 5: Perturbed Maxwellian
  * For 1D/space inhomogenous problem
    * 0: Shock wave (reflection BCs)
    * 1: Sudden heating problem (diffuse reflection BC at one wall, reflection at the other)
    * 2: Two Maxwellian problem (reflection BCs)
    * 3: Heat transfer between two plates (diffuse reflection BCs at each wall)
* `SpaceInhom` - 0 or 1. If 0, this is a 0D-3V problem. If 1, this is a 1D-3V problem. (default: 0)
* `Recompute_weights` - 0 or 1. If set to 1, this tells the code to recompute the convolution weights even if they already exist in the Weights directory. (default: 0)
* `Anisotropic` - 0 or 1. If set to 1, this means that we are solving with an anisotropic cross section (e.g. for plasmas). The code checks to see if the weights exist, if they do not the code exits and reminds you that you need to use the separate `WeightGen_` program to precompute these as it is very intensive calculation. (default: 0)
* `mesh_file` - for 1D problems, this points to the file in `input/` that has information about the physical mesh. See below for specifications for this file. (default: not_set)
* `num_species` - sets the number of species, then species names are listed in order in each line afterwords. For generic nondimensional single species runs, set

num_species <br />
1 <br />
default <br /><br />

If you want to do more, contact me. Mass ratio issues can cause multispecies Boltzmann to get infeasibly expensive in a hurry.

#### Mesh file setup

This expects values in a certain order to set up the 1D / physical space mesh. This sets up blocks/regions, each with a specified uniform spatial grid. The spatial grid is assumed to start with x=0 as its leftmost point.

* first line - comment that is ignored
* second line - total number of points in the mesh
* third line - total number of regions

Next, you define the regions as two lines each
* first line - number of points in the region
* second line - size of the region in (nondimensional) physical space.

For example, a mesh that has a spacing of 0.01 for the first 0.1 of the region and 0.1 for the remaining 0.9 would be


%  Example mesh <br />
19 <br />
2 <br />
10 <br />
0.1 <br />
9 <br />
0.9


#### Ouptut flag file setup

This reads the specified file looking for the following keywords. A 1 in the line following the keyword indicates that you want to dump this information

* `density` - number density at each grid point
* `velocity`- x component of bulk velocity at each grid point
* `temperature` - temperature at each grid point
* `pressure` - pressure at each grid point
* `marginal` - marginal distribution function $\int f dv_2 dv_3$ at each grid point
* `slice` - slice of the distribution function at v_2 = v_3 = N/2
* `entropy` - Boltzmann entropy at each grid point. Negative distribution function values are set to zero.

## Instructions for `WeightGen_`

This generates the precomputed weights for anisotropic cross sections (e.g. Coulomb interactions).

### Setup and run

This is run using

`mpirun -n <number of tasks> WeightGen_ N glance beta lambda L_v`

where

* `N` - the number of grid points in one direction in velocity space / Fourier modes in one direction. The total size of the weights array will be N^6
* `glance` - the angular cutoff for the Coulomb cross section.
  * Set this to 0 to generate weights for the Landau collision operator
* `beta` - parameter for inelastic collisions. Set to 1 for the usual/elastic case
* `lambda` - exponent on the relative velocity in the scattering kernel. Set to `-3` for Coulomb.
* `L_v` - Nondimensional semi-length of velocity domain.
