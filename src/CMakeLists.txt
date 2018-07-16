set(lib_SOURCES
  aniso_weights.c
  boundaryConditions.c
  collisions.c
  collisions_support_cpu.c
  conserve.c
  function_timer.c
  gauss_legendre.c
  initializer.c
  input.c
  mesh_setup.c
  momentRoutines.c
  output.c
  restart.c
  species.c
  transportroutines.c
  weights.c
)

set(lib_CUDA_SOURCES
  collisions_support_gpu.cu
)

add_SpectralBTE_library(lib "${lib_SOURCES}")
add_SpectralBTE_CUDA_sources("${lib_CUDA_SOURCES}")
