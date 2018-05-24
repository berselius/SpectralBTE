#ifndef _OUTPUT_H
#define _OUTPUT_H

#include "species.h"

void initialize_output_hom(int nodes, double Lv, int restart, char *inputFile, char *outputOptions, species *mix, int num_species);

void initialize_output_inhom(int nodes, double Lv, int numX, int numX_node, double *xnodes, double *dxnodes, int restart, char *inputFile, char *outputOptions, species *mix, int num_species);

void write_streams(double **f, double time, double *v);

void write_streams_inhom(double ***f, double time, double *v, int order);

void close_streams();

#endif
