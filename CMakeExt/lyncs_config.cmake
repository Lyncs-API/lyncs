find_package(Python3)
if(NOT Python3_FOUND)
  message( FATAL_ERROR "Python3 needed, CMake will exit." )
endif()

configure_file(setup.py.in setup.py)
add_subdirectory(lyncs_config)
install(CODE "execute_process(COMMAND ${Python3_EXECUTABLE} setup.py install --user OUTPUT_QUIET)")