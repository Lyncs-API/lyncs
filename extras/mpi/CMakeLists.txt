cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(MPI C)

find_package(MPI REQUIRED)

if(NOT MPI_FOUND)
  message(ERROR "MPI not found. Install MPI first or load respective modules.")
endif()

configure_file(config.py.in ${EXTERNAL_INSTALL_LOCATION}/config.py)
