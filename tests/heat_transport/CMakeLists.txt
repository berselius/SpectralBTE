#this is ugly, but we need to be able to call this target in the test command
add_custom_target(heat_transport_input_files
  COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/tests/run_test.sh .
  COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/tests/check_diff.py .
)
add_dependencies(heat_transport_input_files boltz_)

find_program(BASH_PROGRAM bash)
find_program(PYTHON_PROGRAM python)
add_test(NAME heat_transport COMMAND ${BASH_PROGRAM} -c "${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target heat_transport_input_files && ${CMAKE_CURRENT_BINARY_DIR}/run_test.sh heat_transport ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR} $<TARGET_FILE:boltz_> ${PYTHON_PROGRAM}")
