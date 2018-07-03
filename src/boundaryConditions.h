#include "species.h"

void initializeBC(int nv, double *vel, species *mix); 

/*$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/

void setDiffuseReflectionBC(double *in, double *out, double TW, int bdry, int id);
