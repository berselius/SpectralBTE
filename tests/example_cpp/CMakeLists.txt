add_custom_command(
  OUTPUT example_cpp_input_files
  COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/tests/example_cpp/input .
)

add_executable(example_cpp test.cpp example_cpp_input_files)

add_test(example_cpp example_cpp)

target_link_libraries(example_cpp
  SpectralBTE_static
  ${SPECTRALBTE_EXTERNAL_LIBS}
)
