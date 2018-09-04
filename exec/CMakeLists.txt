add_executable(boltz_ boltz.c)
target_link_libraries(boltz_
  SpectralBTE_static
  ${SPECTRALBTE_EXTERNAL_LIBS}
)

# add_executable(weights_ MPIWeightGenerator.c MPIcollisionRoutines.c)
# target_link_libraries(weights_
#   SpectralBTE_static
#   ${SPECTRALBTE_EXTERNAL_LIBS}
# )
