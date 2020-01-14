include(ExternalProject)

set(EXTERNAL_INSTALL_LOCATION ${CMAKE_BINARY_DIR}/external)
if(NOT QUDA_LIBRARY_PATH)
  message(STATUS "No QUDA_LIBRARY_PATH given; we will download it.")
  ExternalProject_Add(quda
    GIT_REPOSITORY https://github.com/lattice/quda
    GIT_TAG develop
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}
  )
  set(QUDA_LIBRARY_PATH ${EXTERNAL_INSTALL_LOCATION})
  set(QUDA_INSTALL ON)
endif()

message(STATUS "Searching for QUDA library")
if(EXISTS "${QUDA_LIBRARY_PATH}/lib/libquda.so" AND EXISTS "${QUDA_LIBRARY_PATH}/include/quda.h")
    set(QUDA_FOUND ON)
endif()

