void initialize_weights_AnIso(int nodes, double *zeta, double lam, double Lv,
                              int weightFlag, double **conv_weights,
                              double glance);

double ghat_theta_AnIso(double theta, void *args);

double ghat_theta2(double theta, void *args);

double ghat_phi_AnIso(double phi, void *args);

double ghat_r_AnIso(double r, void *args);

double gHat3_AnIso(double zeta1, double zeta2, double zeta3, double xi1,
                   double xi2, double xi3);

double ghatL2(double theta, void *args);

double ghatL(double r, void *args);

double gHat3L(double zeta1, double zeta2, double zeta3, double xi1, double xi2,
              double xi3);

void generate_conv_weights_AnIso(double **conv_weights);
