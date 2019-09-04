include(ExternalProject)

if(ENABLE_QUDA)
  if(NOT "${QUDA_LIBRARY_PATH}" STREQUAL "")
    message(STATUS "Searching for QUDA library")
    set(QUDA_FOUND ON)
  else()
    message(STATUS "No QUDA_LIBRARY_PATH provided; we will download it.")
    set(EXTERNAL_INSTALL_LOCATION ${CMAKE_BINARY_DIR}/external)
    ExternalProject_Add(quda
      GIT_REPOSITORY https://github.com/lattice/quda
      GIT_TAG develop
      CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}
    )
    include_directories(${EXTERNAL_INSTALL_LOCATION}/include)
    link_directories(${EXTERNAL_INSTALL_LOCATION}/lib)
  endif ()
endif ()

