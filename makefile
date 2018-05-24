# Directories
DIR=$(PWD)/
EXECDIR=$(DIR)
OBJDIR=$(DIR)obj/
SRCDIR=$(DIR)src/

# GNU C compiler 
CC=mpicc
MPICC=mpicc
WEIGHTCC=mpicc

# Compiler flags
CFLAGS= -O2 -fopenmp -Wall
FFTFLAGS = -lgsl -lgslcblas -lfftw3_omp -lfftw3 -lm

WEIGHTCFLAGS = -O2 -fopenmp -Wall
WEIGHTFFTFLAGS = -lgsl -lgslcblas -lfftw3 -lm

# Command definition
RM=rm -f

# sources for main
sources = $(SRCDIR)main.c $(SRCDIR)initializer.c $(SRCDIR)restart.c

sources_adapt = $(SRCDIR)main_flowadapt.c $(SRCDIR)initializer_flowadapt.c $(SRCDIR)restart.c

sources2 = $(SRCDIR)fractional.c $(SRCDIR)initializer.c $(SRCDIR)restart.c

sources_fast = $(SRCDIR)main_fast.c $(SRCDIR)initializer_fast.c

objects = weights.o collisions.o output.o input.o gauss_legendre.o momentRoutines.o conserve.o transportroutines.o boundaryConditions.o mesh_setup.o species.o pot_weights.o aniso_weights.o poisson.o

objects_adapt = weights.o collisions.o output.o input.o gauss_legendre.o momentRoutines.o conserve.o Flowadapt_transportroutines.o boundaryConditions.o mesh_setup.o species.o pot_weights.o aniso_weights.o flowadapt.o

objects_fast = collisions_fast.o output.o input.o gauss_legendre.o momentRoutines.o conserve.o species.o

weight_sources = $(SRCDIR)MPIWeightGenerator.c $(SRCDIR)MPIcollisionroutines.c $(SRCDIR)gauss_legendre.c

pref_objects = $(addprefix $(OBJDIR), $(objects))

pref_objects_adapt = $(addprefix $(OBJDIR), $(objects_adapt))

pref_objects_fast = $(addprefix $(OBJDIR), $(objects_fast))

# linking step
boltz: $(pref_objects) $(sources)
	@echo "Building Boltzmann deterministic solver"
	$(CC) $(CFLAGS) -o $(EXECDIR)boltz_ $(sources) $(pref_objects) $(FFTFLAGS)
	codesign -f -s haack-boltz $(EXECDIR)boltz_

flowadapt: $(pref_objects_adapt) $(sources_adapt)
	@echo "Building Flow-adapted Boltzmann deterministic solver"
	$(CC) $(CFLAGS) -o $(EXECDIR)Adapt_boltz_ $(sources_adapt) $(pref_objects_adapt) $(FFTFLAGS)

fast: $(pref_objects_fast) $(sources_fast)
	@echo "Building NLogN Boltzmann deterministic solver"
	$(CC) $(CFLAGS) -o $(EXECDIR)Fastboltz_ $(sources_fast) $(pref_objects_fast) $(FFTFLAGS)
	codesign -f -s haack-boltz $(EXECDIR)Fastboltz_

fractional: $(pref_objects) $(sources2)
	@echo "Building Boltzmann deterministic solver with fractional calculus"
	$(CC) $(CFLAGS) -o $(EXECDIR)frac_boltz_ $(sources2) $(pref_objects) $(FFTFLAGS)

weights: $(weight_sources)
	@echo "Building Anisotropic weight generator"
	$(MPICC) $(WEIGHTCFLAGS) -o $(EXECDIR)WeightGen_ $(weight_sources) $(WEIGHTFFTFLAGS)

scatter: $(SRCDIR)pot_main.c $(SRCDIR)scatter.c
	@echo "Buliding cross section tester"
	gcc $(SRCDIR)pot_main.c $(SRCDIR)scatter.c -o scattertest_ -lm -lgsl -lgslcblas

rhostar: $(SRCDIR)pot_rhostar.c $(SRCDIR)scatter.c
	@echo "Buliding rhostar tester"
	gcc $(SRCDIR)pot_rhostar.c $(SRCDIR)scatter.c -o scattertest_ -lm -lgsl -lgslcblas

$(OBJDIR)MPIWeightGenerator.o : $(SRCDIR)MPIWeightgenerator.c
	@echo "Compiling  $< ... " ; \
	if [ -f  $@ ] ; then \
		rm $@ ;\
	fi ; \
	$(WEIGHTCC)  -c $(WEIGHTCFLAGS)  $< -o $@ 2>&1 ;

$(OBJDIR)MPIcollisionroutines.o : $(SRCDIR)MPIcollisionroutines.c
	@echo "Compiling  $< ... " ; \
	if [ -f  $@ ] ; then \
		rm $@ ;\
	fi ; \
	$(WEIGHTCC)  -c $(WEIGHTCFLAGS)  $< -o $@ 2>&1 ;

$(OBJDIR)collisions.o: $(SRCDIR)collisions.c 
	@echo "Compiling  $< ... " ; \
	if [ -f  $@ ] ; then \
		rm $@ ;\
	fi ; \
	$(CC)  -c $(CFLAGS)  $< -o $@ 2>&1 ;

$(OBJDIR)collisions_fast.o: $(SRCDIR)collisions_fast.c 
	@echo "Compiling  $< ... " ; \
	if [ -f  $@ ] ; then \
		rm $@ ;\
	fi ; \
	$(CC)  -c $(CFLAGS)  $< -o $@ 2>&1 ;

$(OBJDIR)output.o: $(SRCDIR)output.c 
	@echo "Compiling  $< ... " ; \
	if [ -f  $@ ] ; then \
		rm $@ ;\
	fi ; \
	$(CC)  -c $(CFLAGS)  $< -o $@ 2>&1 ;

$(OBJDIR)input.o: $(SRCDIR)input.c
	@echo "Compiling  $< ... " ; \
	if [ -f  $@ ] ; then \
		rm $@ ;\
	fi ; \
	$(CC)  -c $(CFLAGS)  $< -o $@ 2>&1 ;

$(OBJDIR)weights.o: $(SRCDIR)weights.c 
	@echo "Compiling  $< ... " ; \
	if [ -f  $@ ] ; then \
		rm $@ ;\
	fi ; \
	$(CC)  -c $(CFLAGS)  $< -o $@ 2>&1 ;

$(OBJDIR)gauss_legendre.o: $(SRCDIR)gauss_legendre.c
	@echo "Compiling  $< ... " ; \
	if [ -f  $@ ] ; then \
		rm $@ ;\
	fi ; \
	$(CC)  -c $(CFLAGS)  $< -o $@ 2>&1 ;

$(OBJDIR)momentRoutines.o : $(SRCDIR)momentRoutines.c
	@echo "Compiling  $< ... " ; \
	if [ -f  $@ ] ; then \
		rm $@ ;\
	fi ; \
	$(CC)  -c $(CFLAGS)  $< -o $@ 2>&1 ;

$(OBJDIR)conserve.o : $(SRCDIR)conserve.c
	@echo "Compiling  $< ... " ; \
	if [ -f  $@ ] ; then \
		rm $@ ;\
	fi ; \
	$(CC)  -c $(CFLAGS)  $< -o $@ 2>&1 ;

$(OBJDIR)transportroutines.o : $(SRCDIR)transportroutines.c
	@echo "Compiling  $< ... " ; \
	if [ -f  $@ ] ; then \
		rm $@ ;\
	fi ; \
	$(CC)  -c $(CFLAGS)  $< -o $@ 2>&1 ;

$(OBJDIR)Flowadapt_transportroutines.o : $(SRCDIR)Flowadapt_transportroutines.c
	@echo "Compiling  $< ... " ; \
	if [ -f  $@ ] ; then \
		rm $@ ;\
	fi ; \
	$(CC)  -c $(CFLAGS)  $< -o $@ 2>&1 ;

$(OBJDIR)flowadapt.o : $(SRCDIR)flowadapt.c
	@echo "Compiling  $< ... " ; \
	if [ -f  $@ ] ; then \
		rm $@ ;\
	fi ; \
	$(CC)  -c $(CFLAGS)  $< -o $@ 2>&1 ;

$(OBJDIR)boundaryConditions.o : $(SRCDIR)boundaryConditions.c
	@echo "Compiling  $< ... " ; \
	if [ -f  $@ ] ; then \
		rm $@ ;\
	fi ; \
	$(CC)  -c $(CFLAGS)  $< -o $@ 2>&1 ;

$(OBJDIR)mesh_setup.o : $(SRCDIR)mesh_setup.c
	@echo "Compiling  $< ... " ; \
	if [ -f  $@ ] ; then \
		rm $@ ;\
	fi ; \
	$(CC)  -c $(CFLAGS)  $< -o $@ 2>&1 ;

$(OBJDIR)species.o : $(SRCDIR)species.c
	@echo "Compiling  $< ... " ; \
	if [ -f  $@ ] ; then \
		rm $@ ;\
	fi ; \
	$(CC)  -c $(CFLAGS)  $< -o $@ 2>&1 ;

$(OBJDIR)pot_weights.o : $(SRCDIR)pot_weights.c
	@echo "Compiling  $< ... " ; \
	if [ -f  $@ ] ; then \
		rm $@ ;\
	fi ; \
	$(CC)  -c $(CFLAGS)  $< -o $@ 2>&1 ;

$(OBJDIR)aniso_weights.o : $(SRCDIR)aniso_weights.c
	@echo "Compiling  $< ... " ; \
	if [ -f  $@ ] ; then \
		rm $@ ;\
	fi ; \
	$(CC)  -c $(CFLAGS)  $< -o $@ 2>&1 ;

$(OBJDIR)poisson.o : $(SRCDIR)poisson.c
	@echo "Compiling  $< ... " ; \
	if [ -f  $@ ] ; then \
		rm $@ ;\
	fi ; \
	$(CC)  -c $(CFLAGS)  $< -o $@ 2>&1 ;


clean:
	$(RM) $(OBJDIR)*.o 
	$(RM) $(EXECDIR)*_

