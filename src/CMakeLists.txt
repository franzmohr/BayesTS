


#####################################################################
# Internal libraries
add_library(bayests_subcommands
    coefficients.cpp
    forecasts.cpp
    loglik.cpp
    posterior.cpp
)

add_library(BayesTS::subcommands ALIAS bayests_subcommands)
target_link_libraries(bayests_subcommands PRIVATE bayests_warnings)

#####################################################################
# Executable
add_executable(bayests bayests.cpp)

target_link_libraries(bayests PRIVATE bayests_warnings)


#####################################################################
# Subdirectories
#
# core first: it owns the structs the io and model layers are written against,
# and it is the only one of them an embedded host has to be able to build.
add_subdirectory(core)
add_subdirectory(io/hdf5)
add_subdirectory(reporters)
add_subdirectory(models)

#####################################################################
# Linking
target_link_libraries(bayests_subcommands PRIVATE bayests_models)
target_link_libraries(bayests_subcommands PRIVATE bayests_hdf5_helper)
target_link_libraries(bayests PRIVATE bayests_subcommands)

# Link OpenMP for multi-threaded operations
if(OpenMP_CXX_FOUND)
    target_link_libraries(bayests_subcommands PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(bayests PRIVATE OpenMP::OpenMP_CXX)
endif()

# main() is the only place that calls openblas_set_num_threads, so the
# definition and the library it needs go on the executable alone. Whether either
# is there at all is decided in the top-level CMakeLists.txt.
target_link_libraries(bayests PRIVATE bayests_openblas)

#####################################################################
# Installation
#
# Only the executable is installed. bayests_subcommands, bayests_models,
# bayests_core, bayests_algorithms, bayests_console_reporter and
# bayests_hdf5_helper are static implementation details that end up inside it.

install(TARGETS bayests
    RUNTIME_DEPENDENCY_SET bayests_runtime_deps
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)

if(BAYESTS_BUNDLE_RUNTIME_DEPS)
    # file(GET_RUNTIME_DEPENDENCIES), which backs the install rule below, does
    # not look at PATH on Windows -- it only searches the directories it is
    # given. Cover the two places the DLLs come from: the compiler's own
    # bin/ (libstdc++, libgomp, libgfortran, libwinpthread) and the bin/ of
    # every prefix a dependency was found in (OpenBLAS, HDF5).
    get_filename_component(_bayests_compiler_dir "${CMAKE_CXX_COMPILER}" DIRECTORY)
    set(_bayests_dep_dirs "${_bayests_compiler_dir}")
    foreach(_prefix IN LISTS CMAKE_PREFIX_PATH)
        list(APPEND _bayests_dep_dirs "${_prefix}/bin")
    endforeach()
    list(APPEND _bayests_dep_dirs ${BAYESTS_RUNTIME_DEP_DIRS})
    list(REMOVE_DUPLICATES _bayests_dep_dirs)

    install(RUNTIME_DEPENDENCY_SET bayests_runtime_deps
        DIRECTORIES ${_bayests_dep_dirs}
        # Windows API sets are virtual: they resolve to nothing on disk, so
        # they have to be dropped before resolution is attempted.
        PRE_EXCLUDE_REGEXES
            [[api-ms-win-.*]]
            [[ext-ms-.*]]
        # Whatever did resolve into the Windows directory belongs to the OS.
        POST_EXCLUDE_REGEXES
            [[.*[/\\][Ss]ystem32[/\\].*\.dll]]
            [[.*[/\\][Ww][Ii][Nn][Dd][Oo][Ww][Ss][/\\].*\.dll]]
        DESTINATION "${CMAKE_INSTALL_BINDIR}"
    )
endif()
