#ifndef _INPUT_H
#define _INPUT_H

void read_input(int *N, double *L_v, double *Kn, double *lambda, double *dt, int *nT, int *order, int *dataFreq, int *restart, double *restart_time, int *initFlag, int *bcFlag, int *homogFlag, int *weightFlag, int *isoFlag, char **meshFile, int *num_species, char ***species_names,char *inputFilename);

void set_default_values(int *N, double *L_v, double *Kn, double *lambda, double *dt, int *nT, int *order, int *dataFreq, int *restart, double *restart_time, int *initFlag, int *bcFlag, int *homogFlag, int *weightFlag, int *num_species, int *isoFlag, char **meshFile);

void check_input(const int *flag);

void read_line(FILE *file, char line[80]);
 
size_t read_int(FILE *file);

double read_double(FILE *file);
 
void read_line_no_adv(FILE *file, char line[80]);

size_t read_int_no_adv(FILE *file);

double read_double_no_adv(FILE *file);

#endif
