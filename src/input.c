#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "input.h"

/*! 
 *  This function reads the input file
 */

void read_input(int *N, double *L_v, double *Kn, double *lambda, double *dt, int *nT, int *order, int *dataFreq, int *restart, double *restart_time, int *initFlag, int *bcFlag, int *homogFlag, int *weightFlag, int *isoFlag, char **meshFile, int *num_species, char ***species_names, char *inputFilename, int *weightgenFlag) {
  int i;
  char   line[80] = {"dummy"};
  char   input_path[100] = {"./input/"};
  FILE  *input_file;

  /*Set input parameters to default values*/
  set_default_values(N, L_v, Kn, lambda, dt, nT, order, dataFreq, restart, restart_time, bcFlag, homogFlag, weightFlag, num_species, isoFlag, meshFile, weightgenFlag);

  strcat(input_path,inputFilename);
  printf("Opening input file %s\n",input_path);
  /*Open input file*/
  input_file = fopen(input_path,"r");  

  if(input_file == NULL) {
    printf("Error - input file not found\n");
    exit(1);
  }

  /*Read input file*/
  while (strcmp(line,"Stop") != 0) {

    read_line(input_file, line);
    
    /*Number of velocity nodes in each dimension*/
    if (strcmp(line,"N") == 0) {
      *N = read_int(input_file);  
    }

    /*Velocity domain semi-length*/
    if (strcmp(line,"L_v") == 0) 
      *L_v = read_double(input_file); 

    /*Knudsen number*/
    if (strcmp(line,"Knudsen") == 0) 
      *Kn = read_double(input_file);

    /*Exponent in cross section*/
    if (strcmp(line,"Lambda") == 0) 
      *lambda = read_double(input_file);
 
    /*Time-step*/
    if (strcmp(line,"Time_step") == 0)
      *dt = read_double(input_file);

    /*Number of time-steps*/
    if (strcmp(line,"Number_of_time_steps") == 0)
      *nT = read_int(input_file);

    /*Order of accuracy of space discretization*/
    if (strcmp(line,"Space_order") == 0)
      *order = read_int(input_file);

    /*Output file writing rate*/
    if (strcmp(line,"Data_writing_frequency") == 0)
      *dataFreq = read_int(input_file);

    /*Flag to activate restart*/
    if (strcmp(line,"Restart") == 0)
      *restart = read_int(input_file);

    /*Time till restart, if needed*/
    if (strcmp(line,"Restart_time") == 0)
      *restart_time = read_double(input_file);

    /*Initial data flag*/
    if (strcmp(line,"Init_field") == 0) {
      *initFlag = read_int(input_file);
    }

    /*Boundary conditions*/
    if (strcmp(line,"Bound_cond") == 0)  
      *bcFlag = read_int(input_file);

    /*Flag for space homogeneous problem*/
    if (strcmp(line,"SpaceInhom") == 0) {
      *homogFlag = read_int(input_file); 
    }

    /*Flag for storing the Fourier weight function*/
    if (strcmp(line,"Recompute_weights") == 0)
      *weightFlag = read_int(input_file);

    /*AnIsotropic*/
    if (strcmp(line,"Anisotropic") == 0)
      *isoFlag = read_int(input_file);

    /*mesh file location*/
    if(strcmp(line,"mesh_file") == 0) {
      read_line(input_file,line);
      strcpy(*meshFile,line);
    }

    /*number of species */
    if(strcmp(line,"num_species") == 0) {
      *num_species = read_int(input_file);
      //get the species names
      *species_names = malloc((*num_species)*sizeof(char *));			      
      for(i=0;i<(int)(*num_species);i++) {
	read_line(input_file,line);
	printf("%d %d %s\n",(int)i,*num_species,line);
	fflush(stdout);
	(*species_names)[i] = malloc(80*sizeof(char));
	strcpy((*species_names)[i],line);
      }
    }

    if (strcmp(line, "generate_weights") == 0) {
      *weightgenFlag = read_int(input_file);
    }
  }

  if((strcmp(*meshFile,"not set") == 0) && (*homogFlag == 1)) {
    printf("Error: please specify the mesh\n");
    exit(1);
  }
    
  printf("done with input file\n");
  fflush(stdout);

  fclose(input_file);
}

/*!
 *  This function sets the input parameters to their defualt values, and sets up flags for a few if they are not set by the input file
 */ 
void set_default_values(int *N, double *L_v, double *Kn, double *lambda, double *dt, int *nT, int *order, int *dataFreq, int *restart, double *restart_time, int *bcFlag, int *homogFlag, int *weightFlag, int *num_species, int *isoFlag, char **meshFile, int *weightgenFlag)
{
  /*Assumes space-homogeneous problem*/
  *homogFlag = 0;

  /*Tells weights routine to check if weights exist with current parameters. If not, regenerate them*/
  *weightFlag = 0;

  /*Number of velocity modes*/
  *N = 16;

  /*Velocity domain semi-length*/
  *L_v = 5.;

  /*Knudsen number*/
  *Kn = 1;

  /*velocity potential exponent*/
  *lambda = 1;

  /*Order of accuracy of space-discretization*/
  *order = 1;

  /*Option for restarting from a previous solution*/
  *restart = 0;

  /*Time until restart (in seconds)*/
  *restart_time = 85500;

  /*Assumes open boundaries*/
  *bcFlag = 0;

  /*timestep - flags that it isn't set*/
  *dt = -1.0;

  /*number of steps*/
  *nT = 1000;

  /*Output update rate*/
  *dataFreq = 10; 

  /*Anisotropic flag set to isotropic collisions for weight calc*/
  *isoFlag = 0;  

  *meshFile = malloc(80*(sizeof(char)));

  *num_species = 1;

  //Flag for no mesh warning?
  strcpy(*meshFile,"not set");

  /*0 = precompute weights, 1 = generate weights on-the-fly*/
  *weightgenFlag = 1;
  
}


/*!
 *  This function checks the return value of the fscanf function while reading the input file
 */ 
void check_input(const int *flag)
{
  if ((*flag) != 1) {
    printf("\n%s\n","Error while reading the input file!");
    printf("%s\n","Please, change the input file...");
    exit(0);
  }
}

/*!
 * This function reads a string of characters from a file
 */ 
void read_line(FILE *file, char line[80])
{
  int read;

  read = fscanf(file,"%s\n",line);
  check_input(&read);
}

/*!
 * This function reads an unsigned integer from a file
 */ 
size_t read_int(FILE *file)
{
  int    read;
  size_t int_value;

  read = fscanf(file,"%lu\n",&int_value);
  check_input(&read);

  return int_value;
}

/*!
 * This function reads a double from a file
 */ 
double read_double(FILE *file)
{
  int    read;
  double double_value;

  read = fscanf(file,"%lf\n",&double_value);
  check_input(&read);

  return double_value;
}

/*!
 * This function reads a string of characters from a file without moving on a new line
 */ 
void read_line_no_adv(FILE *file, char line[80])
{
  int read;

  read = fscanf(file,"%s",line);
  check_input(&read);
}

/*!
 * This function reads an unsigned integer from a file without moving on a new line
 */ 
size_t read_int_no_adv(FILE *file)
{
  int    read;
  size_t int_value;

  read = fscanf(file,"%lu",&int_value);
  check_input(&read);

  return int_value;
}

/*!
 * This function reads a double from a file without moving on a new line
 */ 
double read_double_no_adv(FILE *file)
{
  int    read;
  double double_value;

  read = fscanf(file,"%lf",&double_value);
  check_input(&read);

  return double_value;
}
