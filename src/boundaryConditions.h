#include "species.h"

void initializeBC(int nv, double *vel, species *mix); 

/*$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/

void setDiffuseReflectionBC(double *in, double *out, double TW, double vW, int bdry, int id);
