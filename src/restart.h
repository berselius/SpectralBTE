#include "species.h"

void init_restart(int nXnode, int inorder, int size, int nums, species *mix);

void store_restart(double ***f, int t, char *inputFilename);

void load_restart(double ***f, int *t, char *inputFilename);
