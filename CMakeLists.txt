

# The floor is what the project actually uses, not what the author happens to
# have installed -- a needlessly high one excludes every distribution. What sets
# it, highest first; raise this only when something here does:
#
#   3.25  packagePresets in CMakePresets.json, which is schema version 6
#   3.22  the ENVIRONMENT_MODIFICATION test property, which is how the tests
#         find their DLLs on Windows (test/CMakeLists.txt)
#   3.21  install(... RUNTIME_DEPENDENCY_SET), which bundles those DLLs into a
#         Windows package (src/CMakeLists.txt)
#   3.20  ctest --test-dir, used by every command in CONTRIBUTING.md and CI
#
# Nothing needs CMake 4. Going below 3.22 means giving up the two Windows
# mechanisms above, and below 3.25 means dropping packagePresets and with it
# `cpack --preset release`; `cmake --build --target package` is unaffected.
cmake_minimum_required(VERSION 3.25)


# Configuring in the source root scatters CMakeCache.txt, CMakeFiles/ and the
# generator's own files among the sources, where they shadow every later
# out-of-source configure and have to be picked out by hand to get rid of. The
# check is before project() so it fires without first spending time on compiler
# and Fortran detection.
#
# A binary directory *below* the source directory is fine and is what the
# presets do -- build/bin/<preset>, which .gitignore already covers. Only the
# binary directory being the source directory itself is rejected.
get_filename_component(_bayests_src_real "${CMAKE_SOURCE_DIR}" REALPATH)
get_filename_component(_bayests_bin_real "${CMAKE_BINARY_DIR}" REALPATH)
if(_bayests_src_real STREQUAL _bayests_bin_real)
    message(FATAL_ERROR
        "In-source builds are not allowed.\n"
        "Configure into a subdirectory instead:\n"
        "    cmake --preset default\n"
        "or, without a preset:\n"
        "    cmake -S . -B build/bin/manual\n"
        "\n"
        "This attempt has already written CMakeCache.txt and CMakeFiles/ into\n"
        "${CMAKE_SOURCE_DIR} -- delete both before configuring again.")
endif()
unset(_bayests_src_real)
unset(_bayests_bin_real)


project(BayesTS
    VERSION 0.0.1
    LANGUAGES C CXX Fortran
    DESCRIPTION "Bayesian Time Series Library"
)


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


# Build options
option(BAYESTS_BUILD_DOCS "Add the 'docs' target for Doxygen API documentation" ON)

# The regression harness in test/ pins the RNG and fingerprints a sampler's
# output, so a refactor can be checked against the recorded fixtures.
option(BAYESTS_BUILD_TESTS "Build the regression harnesses in test/" ON)

# -march=native bakes the build machine's instruction set into the binary, so a
# package built with it dies with an illegal instruction on any older CPU. It is
# therefore off by default and worth switching on only for local builds.
option(BAYESTS_NATIVE_ARCH "Optimise for the building machine's CPU (not redistributable)" OFF)

# Windows has no system package manager to pull the runtime in, so a package is
# only useful if it carries its own DLLs. On Unix the loader finds them.
option(BAYESTS_BUNDLE_RUNTIME_DEPS "Copy the runtime libraries the executable needs into the package" ${WIN32})

# Extra places to look for those runtime libraries, for the case where they do
# not sit in the compiler's bin/ or in a CMAKE_PREFIX_PATH entry.
set(BAYESTS_RUNTIME_DEP_DIRS "" CACHE STRING
    "Additional directories to search when bundling runtime libraries")

# # option(BUILD_SHARED_LIBS "Build shared libraries" OFF)
# # option(BAYESTS_BUILD_TESTS "Build tests" ON)
# # option(BAYESTS_BUILD_EXAMPLES "Build examples" ON)

# # Position independent code for static libraries
# set(CMAKE_POSITION_INDEPENDENT_CODE ON)

#####################################################################
# Dependencies

# Use GNUInstallDirs for standard installation directories
include(GNUInstallDirs)

# Find OpenMP for multi-threaded operations
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found - enabling multi-threaded matrix operations")
    add_definitions(-DARMA_USE_OPENMP)
else()
    message(WARNING "OpenMP not found - matrix operations will run single-threaded")
endif()

# # Workaround: Armadillo's config tries to find BLAS/LAPACK but fails with vcpkg
# # Solution: Find OpenBLAS directly and set BLAS variables manually
# find_package(OpenBLAS CONFIG QUIET)
# find_package(lapack CONFIG QUIET)

# if(OpenBLAS_FOUND)
#     # Manually set BLAS/LAPACK variables that Armadillo expects
#     # Must use CACHE to persist across find_package calls
#     set(BLAS_FOUND TRUE CACHE BOOL "BLAS found")
#     set(BLAS_LIBRARIES OpenBLAS::OpenBLAS CACHE STRING "BLAS libraries")
#     set(LAPACK_FOUND TRUE CACHE BOOL "LAPACK found")
#     set(LAPACK_LIBRARIES OpenBLAS::OpenBLAS CACHE STRING "LAPACK libraries")
#     message(STATUS "Using OpenBLAS from vcpkg for BLAS/LAPACK")
# else()
#     message(FATAL_ERROR "OpenBLAS not found via vcpkg. Please install: vcpkg install openblas")
# endif()

find_package(Armadillo CONFIG REQUIRED)

# HDF5 (needed by HighFive)
find_package(hdf5 CONFIG REQUIRED)

# HighFive (header-only library)
# Manual installation - create interface target
# Try multiple sources: env var, cache var, or search CMAKE_PREFIX_PATH
if(DEFINED ENV{HIGHFIVE_DIR})
    set(HIGHFIVE_INCLUDE_DIR "$ENV{HIGHFIVE_DIR}/include" CACHE PATH "HighFive include directory")
elseif(DEFINED HIGHFIVE_DIR)
    set(HIGHFIVE_INCLUDE_DIR "${HIGHFIVE_DIR}/include" CACHE PATH "HighFive include directory")
else()
    # Search in CMAKE_PREFIX_PATH directories
    message(STATUS "Searching for HighFive in CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")
    find_path(HIGHFIVE_INCLUDE_DIR
        NAMES highfive/H5File.hpp
        PATHS ${CMAKE_PREFIX_PATH}
        PATH_SUFFIXES include
        NO_DEFAULT_PATH
    )
    if(NOT HIGHFIVE_INCLUDE_DIR)
        message(FATAL_ERROR "HighFive not found. Current CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}\nSet HIGHFIVE_DIR or add to CMAKE_PREFIX_PATH, or install via: vcpkg install highfive")
    endif()
endif()

if(EXISTS "${HIGHFIVE_INCLUDE_DIR}/highfive/H5File.hpp")
    add_library(HighFive INTERFACE IMPORTED)
    set_target_properties(HighFive PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${HIGHFIVE_INCLUDE_DIR}"
    )
    target_link_libraries(HighFive INTERFACE hdf5::hdf5-shared)
    message(STATUS "Using HighFive from: ${HIGHFIVE_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "HighFive headers not found in ${HIGHFIVE_INCLUDE_DIR}")
endif()


# SuiteSparse in the future
#find_package(SuiteSparse_config CONFIG REQUIRED)

#####################################################################
# Compile options for the numerical libraries
#
# bayests_algorithms and bayests_core hold the same kind of code -- inner loops over
# Armadillo matrices -- and routines move between them as shared pieces are
# factored out of a sampler. One flag set for both keeps that move from
# quietly changing how the code is compiled. Consumers link it PRIVATE: the
# flags describe how a library is built, not how it is used.

#####################################################################
# Include roots
#
# Two of them, and the split is the point:
#
#   include/  is the contract -- the structs a host fills in and the results it
#             gets back. Armadillo and the standard library, nothing else.
#   src/      is everything internal, reached as "core/...", "io/hdf5/..." and
#             so on, so a header's include line says which layer it belongs to.

add_library(bayests_includes INTERFACE)
add_library(BayesTS::includes ALIAS bayests_includes)
target_include_directories(bayests_includes INTERFACE
    "${PROJECT_SOURCE_DIR}/include"
    "${PROJECT_SOURCE_DIR}/src"
)

add_library(bayests_numeric_flags INTERFACE)
add_library(BayesTS::numeric_flags ALIAS bayests_numeric_flags)

# -ffast-math is deliberately NOT here, and should not be added back.
#
# It relaxes IEEE semantics: NaN and Inf checks may be optimised away, and
# results become a function of the compiler version and optimisation level
# rather than of the source. The regression harness in test/ works by pinning
# the RNG and comparing numerical fingerprints, so under -ffast-math a
# mismatch can no longer be attributed to a code change -- which removes the
# only mechanism this project has for telling a refactor from a regression.
# The samplers spend their time in BLAS and LAPACK anyway, which are compiled
# separately and unaffected by flags set here.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(bayests_numeric_flags INTERFACE
        -funroll-loops                  # Unroll loops for better performance
        $<$<CONFIG:Release>:-O3>        # Per-config, so it also holds for multi-config generators
    )

    if(BAYESTS_NATIVE_ARCH)
        target_compile_options(bayests_numeric_flags INTERFACE
            -march=native      # Use all available CPU instructions (AVX, AVX2, FMA, etc.)
            -mtune=native      # Optimize for the current CPU
        )
    endif()
endif()

#####################################################################
# Warnings
#
# Linked PRIVATE by everything this project compiles itself, and by nothing a
# consumer sees: how strictly BayesTS is compiled is not a usage requirement,
# and an embedded host has its own policy -- CRAN, for one, sets the flags.
#
# Armadillo, HDF5 and HighFive arrive through imported targets whose include
# directories are SYSTEM, so their headers stay silent under these.

add_library(bayests_warnings INTERFACE)
add_library(BayesTS::warnings ALIAS bayests_warnings)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(bayests_warnings INTERFACE
        -Wall
        -Wextra
    )
elseif(MSVC)
    target_compile_options(bayests_warnings INTERFACE /W4)
endif()

#####################################################################
# Subdirectories
add_subdirectory(src)

# Documentation is opt-out and never part of ALL, so a missing Doxygen
# installation only costs the 'docs' target, not the build.
if(BAYESTS_BUILD_DOCS)
    add_subdirectory(docs)
endif()


#####################################################################
# Testing
#
# enable_testing() rather than include(CTest): the latter adds a BUILD_TESTING
# option that would compete with BAYESTS_BUILD_TESTS, plus Nightly/Experimental
# targets for a dashboard this project does not have.

if(BAYESTS_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

#####################################################################
# Installation
#
# The executable itself is installed next to its sources, in
# src/CMakeLists.txt. What is left is the paperwork that has to travel with
# any redistributed binary: the licence, plus the two documents that explain
# what the thing is and how to work on it.

install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE"
    "${CMAKE_CURRENT_SOURCE_DIR}/README.md"
    "${CMAKE_CURRENT_SOURCE_DIR}/CONTRIBUTING.md"
    "${CMAKE_CURRENT_SOURCE_DIR}/CHANGELOG.md"
    DESTINATION "${CMAKE_INSTALL_DOCDIR}"
)

#####################################################################
# Packaging
#
# CPack turns the install tree above into a distributable archive:
#
#   cmake --build <binaryDir> --target package     (or: cpack --preset release)
#
# Everything below only describes the package -- it has no effect on a plain
# build or on `cmake --install`.

set(CPACK_PACKAGE_NAME "${PROJECT_NAME}")
set(CPACK_PACKAGE_VENDOR "Franz X. Mohr")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "${PROJECT_DESCRIPTION}")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_INSTALL_DIRECTORY "${PROJECT_NAME}")
set(CPACK_PACKAGE_CHECKSUM SHA256)

# Quote CPack variables in the generated config instead of re-parsing them as
# CMake lists, so paths containing spaces survive.
set(CPACK_VERBATIM_VARIABLES ON)

# BayesTS-0.0.1-Windows-x86_64.zip
set(CPACK_PACKAGE_FILE_NAME
    "${PROJECT_NAME}-${PROJECT_VERSION}-${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")

# NSIS insists on a .txt licence it can render in the click-through page, and
# the repository file has no extension. Copy it into the build tree rather than
# rename the original.
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/LICENSE"
               "${CMAKE_CURRENT_BINARY_DIR}/LICENSE.txt" COPYONLY)
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_BINARY_DIR}/LICENSE.txt")
set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")

# Debug symbols are the point of a Debug build, so only strip elsewhere.
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CPACK_STRIP_FILES ON)
endif()

#####################################################################
# Package generators
#
# Both archive formats are always produced -- they need no tooling beyond
# CMake's own libarchive, and the pair covers both kinds of consumer: a .zip
# that Windows Explorer opens by itself, and a .tar.gz for anything scripted.
# The Windows installer is added only when NSIS is actually present: listing a
# generator whose tool is missing makes the whole `package` target fail.

if(WIN32)
    set(CPACK_GENERATOR "ZIP;TGZ")

    # NSIS is a 32-bit application, so its installer records itself under
    # WOW6432Node and a 64-bit CMake reading the plain key finds nothing. Look
    # under both, and in the default install directories, before giving up --
    # PATH alone misses a stock installation.
    find_program(BAYESTS_MAKENSIS_EXECUTABLE
        NAMES makensis
        HINTS
            [HKEY_LOCAL_MACHINE\\SOFTWARE\\NSIS]
            [HKEY_LOCAL_MACHINE\\SOFTWARE\\WOW6432Node\\NSIS]
        PATHS
            "$ENV{PROGRAMFILES}/NSIS"
            "$ENV{ProgramFiles\(x86\)}/NSIS"
        DOC "NSIS compiler; enables the Windows installer package")

    if(BAYESTS_MAKENSIS_EXECUTABLE)
        list(APPEND CPACK_GENERATOR "NSIS")
        message(STATUS "NSIS found: ${BAYESTS_MAKENSIS_EXECUTABLE} - 'package' also builds an installer")

        # CPack repeats the search on its own and only consults PATH and the
        # 64-bit registry key, so hand it the path we already resolved.
        set(CPACK_NSIS_EXECUTABLE "${BAYESTS_MAKENSIS_EXECUTABLE}")

        set(CPACK_NSIS_PACKAGE_NAME "${PROJECT_NAME} ${PROJECT_VERSION}")
        set(CPACK_NSIS_DISPLAY_NAME "${PROJECT_NAME} ${PROJECT_VERSION}")
        # bayests is a command-line tool and useless unless it can be typed.
        set(CPACK_NSIS_MODIFY_PATH ON)
        set(CPACK_NSIS_ENABLE_UNINSTALL_BEFORE_INSTALL ON)
    else()
        message(STATUS "NSIS not found - 'package' produces a ZIP archive only. "
                       "Install it from https://nsis.sourceforge.io/ to get an installer.")
    endif()
else()
    set(CPACK_GENERATOR "TGZ;ZIP")
endif()

#####################################################################
# Source package
#
#   cpack --config <binaryDir>/CPackSourceConfig.cmake
#
# The ignore list matters more than usual here: test/ alone holds ~140 MB of
# regenerable HDF5 fixtures, and the build tree is a sibling of the sources
# rather than a subdirectory, so it is not automatically out of the way.

set(CPACK_SOURCE_GENERATOR "ZIP;TGZ")
set(CPACK_SOURCE_PACKAGE_FILE_NAME "${PROJECT_NAME}-${PROJECT_VERSION}-src")
# CPACK_VERBATIM_VARIABLES is on, so these reach CPack as written: a single
# backslash escape, not the doubled one an unquoted config would need.
set(CPACK_SOURCE_IGNORE_FILES
    "/\\.git/"
    "/\\.github/"
    "/\\.vscode/"
    "/\\.vs/"
    "/\\.idea/"
    "/\\.cache/"
    "/\\.claude/"
    "/build/"
    "/__cmake_systeminformation/"
    "/out/"
    "/install/"
    "/cmake-build-.*/"
    # Machine-specific compiler and dependency paths.
    "CMakeUserPresets\\.json$"
    # Sampler output and test fixtures; regenerated, never shipped.
    "\\.h5$"
    "\\.hdf5$"
    # Recorded fingerprints. Machine-specific and worthless to anyone else, and
    # a stale one is actively misleading -- see test/baselines/.gitignore.
    "/test/baselines/"
    "\\.swp$"
    "~$"
)

include(CPack)

#####################################################################
# Export
#
# Not provided yet: BayesTS ships an executable, and nothing consumes its
# libraries via find_package(BayesTS). Should that change, the targets need
# INSTALL_INTERFACE include directories and the HighFive interface target
# above has to be reconstructible from the generated config file.

# # Install targets
# install(TARGETS bayests
#     EXPORT BayesTSTargets
#     RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
#     LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
#     ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
# )

# # Export targets for use by other projects
# install(EXPORT BayesTSTargets
#     FILE BayesTSTargets.cmake
#     NAMESPACE BayesTS::
#     DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/BayesTS
# )

# # Create config file
# include(CMakePackageConfigHelpers)
# write_basic_package_version_file(
#     "${CMAKE_CURRENT_BINARY_DIR}/BayesTSConfigVersion.cmake"
#     VERSION ${PROJECT_VERSION}
#     COMPATIBILITY AnyNewerVersion
# )

# configure_package_config_file(
#     "${CMAKE_CURRENT_SOURCE_DIR}/cmake/BayesTSConfig.cmake.in"
#     "${CMAKE_CURRENT_BINARY_DIR}/BayesTSConfig.cmake"
#     INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/BayesTS
# )

# install(FILES
#     "${CMAKE_CURRENT_BINARY_DIR}/BayesTSConfig.cmake"
#     "${CMAKE_CURRENT_BINARY_DIR}/BayesTSConfigVersion.cmake"
#     DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/BayesTS
# )