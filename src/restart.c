#include "restart.h"
#include "species.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

static int rank;
static int nX_Node;
static int order;
static int N;
static int Ns;
static species *mixture;

void init_restart(int nXnode, int inorder, int size, int nums, species *mix) {

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  nX_Node = nXnode;
  order = inorder;
  N = size;
  Ns = nums;
  mixture = mix;
}

void store_restart(double ***f, int t, char *inputFilename) {
  int l, spec;
  ;
  FILE *fidRestart, *fidTime;
  char buffer_restart[100], buffer_time[100];

  // Store time
  if (rank == 0) {
    sprintf(buffer_time, "Restart/%s_time.plt", inputFilename);
    fidTime = fopen(buffer_time, "w");

    printf("Time stored: %d\n", t);
    fwrite(&t, sizeof(int), 1, fidTime);
    if (fflush(fidTime) != 0) {
      printf("Something happened when trying to save the time\n");
      exit(1);
    }
    fclose(fidTime);
  }

  for (spec = 0; spec < Ns; spec++) {
    sprintf(buffer_restart, "Restart/%s_rank%d_%s.plt", inputFilename, rank,
            mixture[spec].name);
    printf("%s\n", buffer_restart);

    fidRestart = fopen(buffer_restart, "w");

    for (l = order; l < nX_Node + order; l++) {
      fwrite(f[spec][l], sizeof(double), N * N * N, fidRestart);
      // printf("Rank %d storing %d\n",rank,l);
      if (fflush(fidRestart) != 0) {
        printf("Something happened when trying to save the pdf\n");
        exit(1);
      }
    }
    fclose(fidRestart);
  }
  printf("Rank %d done storing pdf, waiting\n", rank);

  MPI_Barrier(MPI_COMM_WORLD);
}

void load_restart(double ***f, int *t, char *inputFilename) {
  int l, spec;
  char buffer_restart[100], buffer_time[100];

  FILE *fidRestart, *fidTime;

  for (spec = 0; spec < Ns; spec++) {
    sprintf(buffer_restart, "Restart/%s_rank%d_%s.plt", inputFilename, rank,
            mixture[spec].name);
    printf("%s\n", buffer_restart);

    printf("Loading data %s\n", buffer_restart);

    fidRestart = fopen(buffer_restart, "r");
    printf("%s\n", buffer_restart);
    if (fidRestart == NULL) {
      printf("Error: unable to open restarted file %s\n", buffer_restart);
      exit(1);
    } else
      printf("Opened file\n");

    int readflag;

    for (l = order; l < (nX_Node + order); l++) {
      printf("Node %d loading object %d of %d\n", rank, l, nX_Node);
      readflag = (int)fread(f[spec][l], sizeof(double), N * N * N, fidRestart);
      if (readflag != N * N * N) {
        printf("Error reloading pdf file\n");
        exit(1);
      }
    }
    fclose(fidRestart);
  }

  if (rank == 0) {
    sprintf(buffer_time, "Restart/%s_time.plt", inputFilename);

    printf("loading time\n");
    fidTime = fopen(buffer_time, "r");
    fread(t, sizeof(int), 1, fidTime);
    fclose(fidTime);
    printf("t: %d\n", *t);
  }
}
