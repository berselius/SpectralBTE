add_executable(boltz_ boltz.c)
target_link_libraries(boltz_
  SpectralBTE_static
  ${SPECTRALBTE_EXTERNAL_LIBS}
)


SET_TARGET_PROPERTIES(boltz_ PROPERTIES LINKER_LANGUAGE CXX)

# add_executable(weights_ MPIWeightGenerator.c MPIcollisionRoutines.c)
# target_link_libraries(weights_
#   SpectralBTE_static
#   ${SPECTRALBTE_EXTERNAL_LIBS}
# )
