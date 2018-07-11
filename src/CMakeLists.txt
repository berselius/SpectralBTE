set(lib_SOURCES
  aniso_weights.c
  boundaryConditions.c
  collisions.c
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

add_SpectralBTE_library(lib "${lib_SOURCES}")
