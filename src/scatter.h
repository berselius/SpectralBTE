double distance_xstar(double x, void *params);

double rhostar(double beta);

double distance_rm(double r, void *params);

double compute_rm(double b, double E, double rhostar);

double compute_chi(double b, double E, double rhostar);

double chi_integrand(double t, void *params);

double BornMayer(double r);

double DebyeHuckel(double r);

void initialize_scatter();
