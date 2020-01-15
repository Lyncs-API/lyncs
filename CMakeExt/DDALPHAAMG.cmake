include(ExternalProject)

if(NOT ENABLE_MPI)
  message(ERROR "DDalphaAMG requires MPI")
endif()

if(NOT DDALPHAAMG_PATH)
  message(STATUS "No DDALPHAAMG_PATH provided; we will download it.")
  if(NOT ENABLE_LIME)
    message(ERROR "DDalphaAMG requires lime")
  endif()
  set(EXTERNAL_INSTALL_LOCATION ${CMAKE_BINARY_DIR}/external)
  FILE(GLOB_RECURSE DDALPHAAMG_PATCHES "${CMAKE_SOURCE_DIR}/patches/DDalphaAMG/*.patch")
  ExternalProject_Add(DDalphaAMG
    GIT_REPOSITORY https://github.com/sbacchio/DDalphaAMG
    GIT_TAG master
    PATCH_COMMAND git apply ${DDALPHAAMG_PATCHES}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND make library MPI_C_COMPILER=${MPI_C_COMPILER} MPI_C_COMPILER_FLAGS=${MPI_C_COMPILER_FLAGS} LIMEDIR=${LIME_PATH} && make -j install PREFIX=${EXTERNAL_INSTALL_LOCATION}
    BUILD_IN_SOURCE 1
    INSTALL_COMMAND ""
  )
  if(NOT LIME_FOUND)
    ExternalProject_Add_StepDependencies(DDalphaAMG build c-lime)
  endif()
  set(DDALPHAAMG_PATH ${EXTERNAL_INSTALL_LOCATION})
  set(DDALPHAAMG_INSTALL ON)
endif()

if(EXISTS "${DDALPHAAMG_PATH}/lib/libhmc.so" AND EXISTS "${DDALPHAAMG_PATH}/include/hmc.h")
  set(DDALPHAAMG_FOUND ON)
endif()

