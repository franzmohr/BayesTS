

add_library(bayests_algorithms
            chan_jeliazkov_2009.cpp chan_jeliazkov_2009.h
            kalman_durbin_koopman_2002.cpp kalman_durbin_koopman_2002.h
            stochvol_ksc_1998.cpp stochvol_ksc_1998.h
            stochvol_ocsn_2007.cpp stochvol_ocsn_2007.h
            wishart.cpp wishart.h
            bvs.cpp bvs.h
            ssvs.cpp ssvs.h
            vec_to_var.cpp
            )

add_library(BayesTS::algorithms ALIAS bayests_algorithms)

#####################################################################
# Include directories
target_link_libraries(bayests_algorithms PUBLIC bayests_includes)

#####################################################################
# Performance optimization flags
#
# Shared with the core samplers, so that a routine factored out of a sampler
# into this library is still compiled the same way. Defined in the top-level
# CMakeLists.
target_link_libraries(bayests_algorithms PRIVATE bayests_numeric_flags)
target_link_libraries(bayests_algorithms PRIVATE bayests_warnings)

#####################################################################
# Linking
target_link_libraries(bayests_algorithms PUBLIC armadillo)

# Link OpenMP for multi-threaded matrix operations
if(OpenMP_CXX_FOUND)
    target_link_libraries(bayests_algorithms PUBLIC OpenMP::OpenMP_CXX)
endif()
