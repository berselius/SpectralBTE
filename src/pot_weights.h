
void setup_pot_weights(double Lv, double xmin, double xmax, char *pot_name);

double distance_rm(double r, void *params);

double compute_rm(double b, double E);

double compute_chi(double b, double E);

double chi_integrand(double t, void *params);

double BornMayer(double r);

double DebyeHuckel(double r);

double ghat_b_pot(double b, void* args);

double ghat_phi_pot(double phi, void* args);

double ghat_r_pot(double r, void* args);

double gHat3_pot(double zeta1, double zeta2, double zeta3, double xi1, double xi2, double xi3);

void initialize_weights_pot(int nodes, double *eta, double Lv, int weightFlag, double **conv_weights, char *pot_name);
