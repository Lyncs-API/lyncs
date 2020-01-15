include(ExternalProject)

if(NOT TMLQCD_LIBRARY_PATH)
  message(STATUS "No TMLQCD_LIBRARY_PATH given; we will download it.")
  set(EXTERNAL_INSTALL_LOCATION ${CMAKE_BINARY_DIR}/external)
  ExternalProject_Add(tmLQCD
    GIT_REPOSITORY https://github.com/etmc/tmLQCD
    GIT_TAG master
    CONFIGURE_COMMAND autoconf && ./configure CC=${CMAKE_C_COMPILER} CFLAGS="-std=gnu99 -Wall -pedantic -O3 -ffast-math -fopenmp" --prefix=${EXTERNAL_INSTALL_LOCATION}
    BUILD_COMMAND make -j
    BUILD_IN_SOURCE 1
    INSTALL_COMMAND make -j install
  )
  set(TMLQCD_LIBRARY_PATH ${EXTERNAL_INSTALL_LOCATION})
  set(TMLQCD_INSTALL ON)
endif()

message(STATUS "Searching for tmLQCD library")
if(EXISTS "${TMLQCD_LIBRARY_PATH}/lib/libhmc.so" AND EXISTS "${TMLQCD_LIBRARY_PATH}/include/hmc.h")
  set(TMLQCD_FOUND ON)
endif()

