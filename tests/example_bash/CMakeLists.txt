#this is ugly, but we need to be able to call this target in the test command
add_custom_target(example_bash_input_files
  COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/tests/example_bash/input .
  COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/tests/example_bash/target .
  COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/tests/example_bash/test.sh .
)

find_program(BASH_PROGRAM bash)
add_test(example_bash ${BASH_PROGRAM} -c "${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target example_bash_input_files && ${CMAKE_CURRENT_BINARY_DIR}/test.sh")
