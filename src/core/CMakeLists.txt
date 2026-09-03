

#####################################################################
# Core numerics
#
# Values in, values out. This target deliberately links neither HDF5 nor
# HighFive and contains no console output, no file access and no exit paths
# other than exceptions: everything an embedded host -- an R package, a
# notebook binding -- either cannot provide or is not allowed to let a library
# do on its own. Adding an I/O dependency here is what the layering is meant to
# prevent, so it should be treated as a design change rather than a build fix.

add_library(bayests_core
            spec.cpp
            inputs.cpp
            models/model_support.h
            models/dfm_support.h
            models/dfm_normal_gamma.cpp
            models/dfm_normal_stochvol.cpp
            models/dfm_tvp_gamma.cpp
            models/dfm_tvp_stochvol.cpp
            models/favar_support.h
            models/favar_normal_wishart.cpp
            models/var_normal_gamma.cpp
            models/var_normal_stochvol.cpp
            models/var_normal_wishart.cpp
            models/var_tvp_gamma.cpp
            models/var_tvp_wishart.cpp
            models/var_tvp_stochvol.cpp
            models/vec_support.h
            models/vec_klgs_2010.cpp
            models/vec_normal_gamma.cpp
            models/vec_normal_stochvol.cpp
            models/vec_normal_wishart.cpp
            models/vec_tvp_gamma.cpp
            models/vec_tvp_stochvol.cpp
            models/vec_tvp_wishart.cpp)

# Namespaced alias. Target names are global to a build tree, so the real names
# carry a bayests_ prefix; this is the name a parent project that pulls BayesTS
# in with add_subdirectory() should link against, and it stays stable if the
# underlying target is ever renamed again.
add_library(BayesTS::core ALIAS bayests_core)

#####################################################################
# Subdirectories
add_subdirectory(algorithms)

#####################################################################
# Include directories
target_link_libraries(bayests_core PUBLIC bayests_includes)

#####################################################################
# Performance optimization flags
#
# PRIVATE: these describe how the core is built here, and an embedded host
# compiles the same sources under its own toolchain's rules. An R package, for
# one, may not use -ffast-math at all.
target_link_libraries(bayests_core PRIVATE bayests_numeric_flags)
target_link_libraries(bayests_core PRIVATE bayests_warnings)

#####################################################################
# Linking
target_link_libraries(bayests_core PUBLIC armadillo)
target_link_libraries(bayests_core PUBLIC bayests_algorithms)

# Link OpenMP for multi-threaded matrix operations
if(OpenMP_CXX_FOUND)
    target_link_libraries(bayests_core PUBLIC OpenMP::OpenMP_CXX)
endif()
