/************************************************
weights.h - contains frontend for generating anisotropic weights
**************************************************/

#ifndef _WEIGHTS_H
#define _WEIGHTS_H

#include "species.h"

void alloc_weights(int N, double ****conv_weights, int total_species);

void dealloc_weights(int N, double **conv_weights);

/*******************
function initialize_weights
This function fills the weight array allocated by alloc_weights. 
Inputs
======
nodes: number of nodes in each direction in velocity space
eta: grid points in Fourier space
Lv: Semi-length of velocity space domain [-L,L]
lam: exponent on the relative velocity term in collision kernel. 1: Hard Spheres, 0: Maxwell, -3:Coulomb
weightFlag: 
===
  If set to 0, the program checks to see if weights have already been stored with these parameters and loads them if so, builds them if not
  If set to 1, the program is forced to rebuild the weight array whether or not it has been precomputed
isoFlag
===
  If set to 0, the program computes assuming an isotropic angular cross section
  If set to 1, the program computes assuming a Landau-Fokker-Planck cross section (generated separately by MPIWeightgenerator)
  (WORK IN PROGRESS - I WILL SLOT THE MORE GENERAL ANISO CROSS SECTION HERE EVENTUALLY)
conv_weights: pointer to memory allocated by alloc_weights
*******************/
void initialize_weights(int lower, int range, int nodes, double *eta, double Lv, double lam, int weightFlag, int isoFlag, double **conv_weights, species species_i, species species_j, int rank, int size);

void write_weights(double **conv_weights, char buffer_weights[100], int N,int lower, int range, int rank, int size);
#endif
