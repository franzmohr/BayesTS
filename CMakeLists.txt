

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

# Which target the HDF5 config exports is not part of any contract: it depends on
# how that HDF5 was built and by whom. Two things vary independently --
#
#   * the link type. A CMake-built HDF5 exports `-shared` and `-static` variants
#     and may install one or both; a hand-built 2.0 installs both, MSYS2's package
#     is static only.
#   * the namespace. Some packagings export `hdf5::hdf5-static`, others the bare
#     `hdf5-static` with no namespace at all -- MSYS2 again.
#
# So the name is resolved once, here, rather than written into each of the three
# places that link it, where a mismatch surfaces as three identical "target was
# not found" errors that name no package.
#
# Shared before static, and namespaced before bare, so a build against an HDF5
# offering several keeps linking the one it always did. An HDF5 that offers only a
# static library is fine: it just means nothing needs bundling next to the
# executable on Windows.
set(BAYESTS_HDF5_CANDIDATE_TARGETS
    hdf5::hdf5-shared hdf5-shared
    hdf5::hdf5        hdf5
    hdf5::hdf5-static hdf5-static)

set(BAYESTS_HDF5_TARGET "")
foreach(_bayests_hdf5_candidate IN LISTS BAYESTS_HDF5_CANDIDATE_TARGETS)
    if(TARGET ${_bayests_hdf5_candidate})
        set(BAYESTS_HDF5_TARGET ${_bayests_hdf5_candidate})
        break()
    endif()
endforeach()
unset(_bayests_hdf5_candidate)

if(NOT BAYESTS_HDF5_TARGET)
    # Report what this HDF5 actually offers, so adding the missing name is one
    # edit rather than a guess. IMPORTED_TARGETS is a directory property, so it
    # lists everything the find_package above defined in this scope. The candidate
    # list is printed from the variable the loop uses, so the two cannot drift.
    get_property(_bayests_imported DIRECTORY PROPERTY IMPORTED_TARGETS)
    list(FILTER _bayests_imported INCLUDE REGEX "(hdf5|HDF5)")
    message(FATAL_ERROR
        "find_package(hdf5) succeeded but exported none of the targets this "
        "project knows how to link:\n"
        "    ${BAYESTS_HDF5_CANDIDATE_TARGETS}\n"
        "HDF5 version: ${HDF5_VERSION}\n"
        "HDF5-related imported targets that do exist:\n"
        "    ${_bayests_imported}\n"
        "Add the right name to BAYESTS_HDF5_CANDIDATE_TARGETS in CMakeLists.txt.")
endif()

message(STATUS "HDF5 target: ${BAYESTS_HDF5_TARGET}")

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
    target_link_libraries(HighFive INTERFACE ${BAYESTS_HDF5_TARGET})
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

        # NSIS_MAX_STRLEN is compiled into makensis and into the stubs it
        # embeds, so it is a property of the machine building the installer,
        # not of the machine running it. The large-strings distribution that
        # raises it is a zip unpacked over an ordinary install, which means
        # any NSIS update silently restores the 1024-character default. The
        # PATH code below is written to be correct at either limit -- report
        # which one is in force so a reversion is at least visible. Note that
        # execute_process spawns makensis directly, so /HDRINFO reaches it as
        # a switch and not as a path some shell has helpfully rewritten.
        execute_process(
            COMMAND "${BAYESTS_MAKENSIS_EXECUTABLE}" /HDRINFO
            OUTPUT_VARIABLE _bayests_hdrinfo
            ERROR_VARIABLE _bayests_hdrinfo
            RESULT_VARIABLE _bayests_hdrinfo_status)
        if(_bayests_hdrinfo_status EQUAL 0
           AND _bayests_hdrinfo MATCHES "NSIS_MAX_STRLEN=([0-9]+)")
            if(CMAKE_MATCH_1 GREATER 1024)
                set(_bayests_strlen_kind "large-strings build")
            else()
                set(_bayests_strlen_kind "stock build")
            endif()
            message(STATUS "NSIS string limit: ${CMAKE_MATCH_1} characters"
                           " - ${_bayests_strlen_kind}")
        endif()

        # CPack repeats the search on its own and only consults PATH and the
        # 64-bit registry key, so hand it the path we already resolved.
        set(CPACK_NSIS_EXECUTABLE "${BAYESTS_MAKENSIS_EXECUTABLE}")

        set(CPACK_NSIS_PACKAGE_NAME "${PROJECT_NAME} ${PROJECT_VERSION}")
        set(CPACK_NSIS_DISPLAY_NAME "${PROJECT_NAME} ${PROJECT_VERSION}")
        set(CPACK_NSIS_ENABLE_UNINSTALL_BEFORE_INSTALL ON)

        # bayests is a command-line tool and useless unless it can be typed,
        # so the installer asks whether to put <install dir>\bin on PATH, on a
        # page of its own offering "for all users", "for the current user" and
        # "do not add". Both that page and the code behind it come from here.
        set(CPACK_NSIS_MODIFY_PATH ON)

        # The AddToPath function CPack generates opens with
        #
        #     ReadEnvStr $1 PATH
        #
        # and stock makensis hands back an empty string for anything longer
        # than NSIS_MAX_STRLEN (1024 characters). On a machine whose merged
        # machine+user PATH is over that -- forty-odd entries will do it --
        # the read comes back empty and the installer ends with "Warning! PATH
        # too long installer unable to modify PATH!", leaving PATH untouched.
        #
        # That read feeds only the "is the directory already listed?" test;
        # every read and write that matters goes to the registry hive picked
        # on the page. So point the test at that same hive: its raw,
        # unexpanded value is far shorter than the merged PATH, and it is the
        # string being rewritten anyway. The length check stays, now measuring
        # what is actually about to be written.
        #
        # CPack resolves its NSIS templates through CPACK_MODULE_PATH before
        # falling back to the copies shipped with CMake, so patch copies at
        # configure time rather than vendor a thousand lines that would then
        # have to track CMake releases. It searches the module path for the
        # bare file names, so the patched copies sit at the top of this
        # directory. Clear it first: a patch that stops applying must not
        # leave last configure's copy behind to be picked up anyway.
        set(_bayests_nsis_dir "${CMAKE_CURRENT_BINARY_DIR}/CPackNSIS")
        file(REMOVE_RECURSE "${_bayests_nsis_dir}")
        set(CPACK_MODULE_PATH ${CMAKE_MODULE_PATH} "${_bayests_nsis_dir}")

        set(_bayests_nsis_stock
            "${CMAKE_ROOT}/Modules/Internal/CPack/NSIS.template.in")
        set(_bayests_nsis_before [=[  ReadEnvStr $1 PATH
  ; if the path is too long for a NSIS variable NSIS will return a 0
  ; length string.  If we find that, then warn and skip any path
  ; modification as it will trash the existing path.
  StrLen $2 $1
  IntCmp $2 0 CheckPathLength_ShowPathWarning CheckPathLength_Done CheckPathLength_Done]=])
        set(_bayests_nsis_after [=[  ; Patched by the BayesTS build. Upstream reads the expanded PATH out of
  ; the environment block here, which makensis truncates to nothing past
  ; NSIS_MAX_STRLEN and which is not the string this function rewrites.
  ; Read whichever hive the options page selected -- the one the writes
  ; further down target -- and warn only if the result would not fit.
  StrCmp $ADD_TO_PATH_ALL_USERS "1" 0 +3
    ReadRegStr $1 ${NT_all_env} "PATH"
    Goto +2
    ReadRegStr $1 ${NT_current_env} "PATH"
  StrLen $2 $1
  StrLen $3 "$0;"
  IntOp $2 $2 + $3
  IntCmp $2 ${NSIS_MAX_STRLEN} CheckPathLength_ShowPathWarning CheckPathLength_Done CheckPathLength_ShowPathWarning]=])

        if(NOT EXISTS "${_bayests_nsis_stock}")
            message(WARNING
                "CPack's NSIS template is not at ${_bayests_nsis_stock} - the "
                "installer will refuse to edit a PATH over 1024 characters.")
        else()
            file(READ "${_bayests_nsis_stock}" _bayests_nsis_text)
            string(FIND "${_bayests_nsis_text}" "${_bayests_nsis_before}"
                   _bayests_nsis_hit)
            if(_bayests_nsis_hit LESS 0)
                message(WARNING
                    "CPack's NSIS template no longer contains the PATH-length "
                    "check this project patches - the installer will refuse to "
                    "edit a PATH over 1024 characters.")
            else()
                string(REPLACE "${_bayests_nsis_before}" "${_bayests_nsis_after}"
                       _bayests_nsis_text "${_bayests_nsis_text}")
                file(WRITE "${_bayests_nsis_dir}/NSIS.template.in"
                     "${_bayests_nsis_text}")
            endif()
        endif()

        # The page starts on "Do not add", so a user who clicks straight
        # through gets no PATH entry and a bayests that cannot be typed.
        # Start on "for current user" instead: it is the one option that
        # needs no elevation, and it writes to the same hive an unprivileged
        # install puts its files under. "For all users" stays one click away.
        #
        # Field 2 is "do not add" and field 4 is "for current user"; their
        # Top/Bottom coordinates tell those State lines apart from the three
        # others in the file. The \r? is not optional-by-accident -- this
        # template is CRLF where NSIS.template.in is LF.
        set(_bayests_ini_stock
            "${CMAKE_ROOT}/Modules/Internal/CPack/NSIS.InstallOptions.ini.in")
        if(NOT EXISTS "${_bayests_ini_stock}")
            message(WARNING
                "CPack's NSIS options page is not at ${_bayests_ini_stock} - "
                "the installer will default to leaving PATH alone.")
        else()
            file(READ "${_bayests_ini_stock}" _bayests_ini)
            string(REGEX REPLACE "(Top=30\r?\nBottom=40\r?\n)State=1"
                   "\\1State=0" _bayests_ini "${_bayests_ini}")
            string(REGEX REPLACE "(Top=50\r?\nBottom=60\r?\n)State=0"
                   "\\1State=1" _bayests_ini "${_bayests_ini}")
            # The label above the buttons announces the old default.
            string(REPLACE
                "By default @CPACK_PACKAGE_INSTALL_DIRECTORY@ does not add its directory to the system PATH."
                "@CPACK_PACKAGE_NAME@ is a command-line tool. Its directory has to be on the PATH to run it from anywhere."
                _bayests_ini "${_bayests_ini}")
            if(_bayests_ini MATCHES "Top=50\r?\nBottom=60\r?\nState=1")
                file(WRITE "${_bayests_nsis_dir}/NSIS.InstallOptions.ini.in"
                     "${_bayests_ini}")
            else()
                message(WARNING
                    "CPack's NSIS options page no longer has the radio buttons "
                    "this project re-orders - the installer will default to "
                    "leaving PATH alone.")
            endif()
        endif()
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