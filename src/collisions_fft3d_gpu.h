#ifndef _COLLISIONS_FFT3D_GPU_H
#define _COLLISIONS_FFT3D_GPU_H

void initialize_collisions_support_gpu(const double *wtN_global, const int num);

void deallocate_collisions_support_gpu();

void fft3D_gpu(const double (*in)[2], double (*out)[2], const double delta, const double L_start, const double L_end, const double sign, const double *var, const double scaling, const int N);

#endif
