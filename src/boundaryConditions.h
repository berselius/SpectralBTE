#include "species.h"

void initializeBC(int nv, double *vel, species *mix); 


void initializeBC_shock(int nv, double *vel, species *mix, int n_l, int n_r, double u_l, double u_r, double T_l, double T_r);

/*$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/

void setDiffuseReflectionBC(double *in, double *out, double TW, int bdry, int id);

void setMaxwellBC(double *out, int bdry, int id);
