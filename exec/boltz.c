#include <fftw3.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "initializer.h"
#include "input.h"
#include "mesh_setup.h"
#include "output.h"
#include "restart.h"

#include "collisions.h"
#include "conserve.h"
#include "species.h"
#include "transportroutines.h"

#include "weights.h"
//#include "pot_weights.h"
#include "aniso_weights.h"

int main(int argc, char **argv) {
  //**********************
  // MPI stuff
  //**********************
  int rank;
  int numNodes;
  int nX_Node;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numNodes);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Top-level parameters, read from input
  int N;
  double L_v;
  double Kn;
  double lambda;
  double dt;
  int nT;
  int order;
  int dataFreq;
  int restart;
  double restart_time;
  int initFlag;
  int bcFlag;
  int homogFlag;
  int weightFlag;
  int isoFlag;
  char *meshFile = malloc(80 * sizeof(char));
  int weightgen = 0; // 0 = generate weights before, 1 = on the fly

  // parameters for the spatial mesh, if doing inhomogeneous
  int nX;
  double *x, *dx, dx_min;

  // other top level parameters
  double *v;
  double *zeta;
  double **f_hom;    // levels - species, v-position
  double **f_hom1;   // levels - species, v-position
  double ***f_inhom; // levels - species, x-postion, v-position
  double ***f_conv; // levels - species, x-postion, v-position (for intermediate
                    // step in splitting)
  double ***f_1; // levels - species, x-postion, v-position (for intermediate
                 // step in 2nd order time integrator)
  double *
      *Q; // array of collision operator results - levels - species, v-position
  double ***conv_weights; // array of convolution weights - levels - interaction
                          // index, zeta, xi

  // Species data
  int num_species;
  species *mixture;
  char **species_names;

  // timing stuff for restart
  double totTime, totTime_start;
  double writeTime, writeTime_start;
  int restart_flag;
  int weight_file_specified = 0;

  // command line arguments
  char *inputFilename = malloc(80 * sizeof(char));
  strcpy(inputFilename, argv[1]);

  char *outputChoices = malloc(80 * sizeof(char));
  strcpy(outputChoices, argv[2]);

  char *weightFilename = malloc(80 * sizeof(char));
  if (argc == 4) {
    weight_file_specified = 1;
    strcpy(weightFilename, argv[3]);
    printf("The weight file is specified to be %s\n", weightFilename);
  }

  // Variables for main function
  int t;
  int i, j, k, l, m, n;
  int outputCount;

  double t1, t2;

  if ((restart_time > 0) && (rank == 0)) {
    totTime_start = MPI_Wtime();
    totTime = 0;
  }

  // load data from input file
  read_input(&N, &L_v, &Kn, &lambda, &dt, &nT, &order, &dataFreq, &restart,
             &restart_time, &initFlag, &bcFlag, &homogFlag, &weightFlag,
             &isoFlag, &meshFile, &num_species, &species_names, inputFilename);

  //////////////////////////////////
  // SETUP                         //
  //////////////////////////////////

  // Allocating and initializing

  load_and_allocate_spec(&mixture, num_species, species_names);

  if (homogFlag == 1) { // load mesh data
    if (rank == 0)
      printf("Loading mesh\n");
    make_mesh(&nX, &nX_Node, &dx_min, &x, &dx, order, meshFile);
  }

  if (rank == 0)
    printf("Initializing variables %d\n", homogFlag);

  if (homogFlag == 0) { // homogeneous case
    allocate_hom(N, &v, &zeta, &f_hom, &f_hom1, &Q, num_species);
    initialize_hom(N, L_v, v, zeta, f_hom, initFlag, mixture);
    t = 0;
  } else { // inhomogeneous case
    allocate_inhom(N, nX_Node + (2 * order), &v, &zeta, &f_inhom, &f_conv, &f_1,
                   &Q, num_species);
    initialize_inhom(N, num_species, L_v, v, zeta, f_inhom, f_conv, f_1,
                     mixture, initFlag, nX_Node, x, dx, dt, &t, order, restart,
                     inputFilename);
  }

  // Setup output

  if (rank == 0)
    printf("Initializing output %s\n", outputChoices);
  if (homogFlag == 0)
    initialize_output_hom(N, L_v, restart, inputFilename, outputChoices,
                          mixture, v, num_species);
  else
    initialize_output_inhom(N, L_v, nX, nX_Node, x, dx, restart, inputFilename,
                            outputChoices, mixture, num_species);

  // Setup weights

  if (rank == 0)
    printf("Initializing weight info\n");

  if (weight_file_specified) {
    FILE *fidWeights;
    char buffer_weights[256];
    int readFlag;

    if (num_species > 1) {
      printf(
          "Error - weight file loading only implemented for single species\n");
      exit(1);
    }

    printf("Loading weights stored in %s \n", weightFilename);
    sprintf(buffer_weights, "Weights/%s", weightFilename);

    alloc_weights(N, &conv_weights, 1);

    if ((fidWeights = fopen(buffer_weights, "r"))) {
      printf("Opened %s\n", buffer_weights);
      for (i = 0; i < N * N * N; i++) {
        conv_weights[0][i] = malloc(N * N * N * sizeof(double));
        readFlag = (int)fread(conv_weights[0][i], sizeof(double), N * N * N,
                              fidWeights);
        if (readFlag != N * N * N) {
          printf("Readflag %d\n", readFlag);
          printf("Error reading weight file\n");
          exit(1);
        }
      }
      fclose(fidWeights);
    } else {
      printf("Error opening weight file %s, please check the name\n",
             weightFilename);
      exit(1);
    }
  }

  else {

    if (weightgen == 0) {
      printf("Preparing for precomputation of weights...\n");
      alloc_weights(N, &conv_weights, num_species * num_species);

      if (isoFlag == 1) { // load AnIso weights generated by the AnIso weight
                          // generator (seperate program)
        initialize_weights_AnIso(N, zeta, L_v, lambda, weightFlag,
                                 conv_weights[0], 0.0);
      } else {
        for (i = 0; i < num_species; i++)
          for (j = 0; j < num_species; j++)
            initialize_weights(N, zeta, L_v, lambda, weightFlag, isoFlag,
                               conv_weights[j * num_species + i], mixture[i],
                               mixture[j]);
      }
    } else {
      printf("Not precomputing weights; The weights will be computed on the "
             "fly...\n");
    }
  }

  outputCount = 0;

  // Set up conservation routines

  initialize_conservation(N, v[1] - v[0], v, mixture, num_species);

  //////////////////////////////////////////////////
  // ALL SETUP COMPLETE                            //
  //////////////////////////////////////////////////

  printf("Done with all setup, starting main loop\n");

  fflush(stdout);

  writeTime_start = MPI_Wtime();
  totTime_start = MPI_Wtime();

  //////////////////////////////////////////////
  // SPACE HOMOGENEOUS CASE                    //
  //////////////////////////////////////////////
  if (homogFlag == 0) {
    write_streams(f_hom, 0);

    while (t < nT) {
      printf("In step %d of %d\n", t + 1, nT);
      t1 = omp_get_wtime();

      for (l = 0; l < num_species; l++) {
        for (m = 0; m < num_species; m++) {
          ComputeQ(f_hom[l], f_hom[m], Q[m * num_species + l],
                   conv_weights[m * num_species + l]);
          // ComputeQ_maxPreserve(f_hom[l],f_hom[m],Q[m*num_species +
          // l],conv_weights[m*num_species + l]);
        }
      }

      // conserve
      conserveAllMoments(Q);

      t2 = omp_get_wtime();
      printf("Time elapsed: %g\n", t2 - t1);

      if (order == 1) {
        // update
        for (i = 0; i < N * N * N; i++)
          for (l = 0; l < num_species; l++)
            for (m = 0; m < num_species; m++)
              f_hom[l][i] += dt * Q[m * num_species + l][i] / Kn;
      } else if (order == 2) {
        // update

        for (i = 0; i < N * N * N; i++)
          for (l = 0; l < num_species; l++) {
            f_hom1[l][i] = f_hom[l][i];
            for (m = 0; m < num_species; m++)
              f_hom1[l][i] += dt * Q[m * num_species + l][i] / Kn;
          }

        // compute new collision ops
        for (l = 0; l < num_species; l++) {
          for (m = 0; m < num_species; m++) {
            ComputeQ(f_hom1[l], f_hom1[m], Q[m * num_species + l],
                     conv_weights[m * num_species + l]);
            // ComputeQ_maxPreserve(f_hom1[l],f_hom1[m],Q[m*num_species +
            // l],conv_weights[m*num_species + l]);
          }
        }

        // conserve
        conserveAllMoments(Q);

        // update 2
        for (i = 0; i < N * N * N; i++)
          for (l = 0; l < num_species; l++) {
            f_hom[l][i] = 0.5 * (f_hom[l][i] + f_hom1[l][i]);
            for (m = 0; m < num_species; m++)
              f_hom[l][i] += 0.5 * dt * Q[m * num_species + l][i] / Kn;
          }
      }

      outputCount++;
      if (outputCount % dataFreq == 0) {
        write_streams(f_hom, dt * (t + 1));
        outputCount = 0;
      }
      t = t + 1;
    }
  }
  /////////////////////////////////////////////
  // SPACE INHOMOGENEOUS CASE                 //
  /////////////////////////////////////////////
  else {
    if (!restart)
      write_streams_inhom(f_inhom, 0, order);

    if ((rank == 0) && (restart_time > 0)) {
      totTime = MPI_Wtime() - totTime_start;
      writeTime_start = MPI_Wtime();
      // printf("%g\n",totTime);
    }

    while (t < nT) {
      if (rank == 0)
        printf("In step %d of %d\n", t + 1, nT);

      ///////////////////////////
      // ADVECTION STEP         //
      ///////////////////////////
      for (m = 0; m < num_species; m++) {
        if (order == 1)
          advectOne(f_inhom[m], f_conv[m], m);
        else
          advectTwo(f_inhom[m], f_conv[m], m);
      }

      t1 = (double)clock() / (double)CLOCKS_PER_SEC;

      ////////////////////////
      // COLLISION STEP      //
      ////////////////////////
      for (l = order; l < (nX_Node + order); l++) {
        for (m = 0; m < num_species; m++)
          for (n = 0; n < num_species; n++) {
            ComputeQ(f_conv[m][l], f_conv[n][l], Q[n * num_species + m],
                     conv_weights[n * num_species + m]);
          }

        conserveAllMoments(Q);

        t2 = (double)clock() / (double)CLOCKS_PER_SEC;
        printf("Time elapsed: %g\n", t2 - t1);

        for (m = 0; m < num_species; m++) {

          if (order == 1)
            for (i = 0; i < N; i++)
              for (j = 0; j < N; j++)
                for (k = 0; k < N; k++)
                  f_inhom[m][l][k + N * (j + N * i)] =
                      f_conv[m][l][k + N * (j + N * i)];
          else
            for (i = 0; i < N; i++)
              for (j = 0; j < N; j++)
                for (k = 0; k < N; k++)
                  f_1[m][l][k + N * (j + N * i)] =
                      f_conv[m][l][k + N * (j + N * i)];

          for (n = 0; n < num_species; n++)
            for (i = 0; i < N; i++)
              for (j = 0; j < N; j++)
                for (k = 0; k < N; k++) {

                  if (order == 1)
                    f_inhom[m][l][k + N * (j + N * i)] +=
                        dt * Q[n * num_species + m][k + N * (j + N * i)] / Kn;
                  else
                    f_1[m][l][k + N * (j + N * i)] +=
                        dt * Q[n * num_species + m][k + N * (j + N * i)] / Kn;
                }
        }

        // The RK2 Part
        if (order == 2) {

          for (m = 0; m < num_species; m++)
            for (n = 0; n < num_species; n++)
              ComputeQ(f_1[m][l], f_1[n][l], Q[n * num_species + m],
                       conv_weights[n * num_species + m]);

          conserveAllMoments(Q);

          for (m = 0; m < num_species; m++) {

            for (i = 0; i < N; i++)
              for (j = 0; j < N; j++)
                for (k = 0; k < N; k++)
                  f_conv[m][l][k + N * (j + N * i)] =
                      0.5 * f_conv[m][l][k + N * (j + N * i)] +
                      0.5 * f_1[m][l][k + N * (j + N * i)];

            for (n = 0; n < num_species; n++)
              for (i = 0; i < N; i++)
                for (j = 0; j < N; j++)
                  for (k = 0; k < N; k++)
                    f_conv[m][l][k + N * (j + N * i)] +=
                        0.5 * dt * Q[n * num_species + m][k + N * (j + N * i)] /
                        Kn;
          }
        }
      }

      ////////////////////////////////
      // ADVECTION STEP (IF NEEDED)  //
      ////////////////////////////////
      if (order == 2)
        for (m = 0; m < num_species; m++)
          advectTwo(f_conv[m], f_inhom[m], m);

      ////////////////////////////////
      // RECORD DATA                 //
      ////////////////////////////////
      outputCount++;
      if (outputCount % dataFreq == 0) {
        if (restart_time > 0) {
          if ((rank == 0) && (restart_time > 0)) {
            writeTime = MPI_Wtime() - writeTime_start;
            totTime = MPI_Wtime() - totTime_start;
            writeTime_start = MPI_Wtime();
            // printf("%g %g\n",totTime,writeTime);
            if ((totTime + writeTime) > 0.95 * (double)restart_time) {
              printf(
                  "RESTART TIME REACHED - STORING CURRENT DISTRIBUTION DATA\n");
              restart_flag = 1;
              MPI_Bcast(&restart_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
              store_restart(f_inhom, t, inputFilename);
            } else {
              restart_flag = 0;
              MPI_Bcast(&restart_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
            }
          } else {
            MPI_Bcast(&restart_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if (restart_flag == 1)
              store_restart(f_inhom, t, inputFilename);
          }
          if (restart_flag == 1) {
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Finalize();
            return 0;
          }
        }
        write_streams_inhom(f_inhom, dt * (t + 1), order);
        outputCount = 0;
      }

      t = t + 1;
    }
  }

  if (!homogFlag) { // Homogeneous case
    dealloc_hom(v, zeta, f_hom, Q);
  } else
    dealloc_inhom(nX_Node, order, v, zeta, f_inhom, f_conv, f_1, Q);

  //////////////////////////////////////
  // DEALLOCATION AND WRAP UP          //
  //////////////////////////////////////
  if (rank == 0)
    printf("Wrapping up\n");

  if (rank == 0)
    close_streams(homogFlag);

  if (homogFlag)
    dealloc_trans();

  for (i = 0; i < num_species; i++)
    dealloc_weights(N, conv_weights[i]);
  free(conv_weights);

  dealloc_coll();
  dealloc_conservation();
  MPI_Finalize();

  return 0;
}
