

#####################################################################
# HDF5 front-end
#
# Translates between the model files on disk and the plain structs in
# include/bayests. Nothing above this layer should need HighFive, and nothing
# below it is allowed to.

add_library(bayests_hdf5_helper
            hdf5_and_armadillo.cpp hdf5_and_armadillo.h
            model_io_common.cpp model_io_common.h
            dfm_normal_gamma_io.cpp dfm_normal_gamma_io.h
            dfm_normal_stochvol_io.cpp dfm_normal_stochvol_io.h
            dfm_tvp_gamma_io.cpp dfm_tvp_gamma_io.h
            dfm_tvp_stochvol_io.cpp dfm_tvp_stochvol_io.h
            var_normal_gamma_io.cpp var_normal_gamma_io.h
            var_normal_stochvol_io.cpp var_normal_stochvol_io.h
            var_normal_wishart_io.cpp var_normal_wishart_io.h
            var_tvp_gamma_io.cpp var_tvp_gamma_io.h
            var_tvp_wishart_io.cpp var_tvp_wishart_io.h
            var_tvp_stochvol_io.cpp var_tvp_stochvol_io.h
            vec_klgs_2010_io.cpp vec_klgs_2010_io.h
            vec_normal_gamma_io.cpp vec_normal_gamma_io.h
            vec_normal_stochvol_io.cpp vec_normal_stochvol_io.h
            vec_normal_wishart_io.cpp vec_normal_wishart_io.h
            vec_tvp_gamma_io.cpp vec_tvp_gamma_io.h
            vec_tvp_stochvol_io.cpp vec_tvp_stochvol_io.h
            vec_tvp_wishart_io.cpp vec_tvp_wishart_io.h)

add_library(BayesTS::hdf5_helper ALIAS bayests_hdf5_helper)
target_link_libraries(bayests_hdf5_helper PRIVATE bayests_warnings)

#####################################################################
# Include directories
target_link_libraries(bayests_hdf5_helper PUBLIC bayests_includes)

#####################################################################
# Linking
target_link_libraries(bayests_hdf5_helper PUBLIC armadillo)
# Resolved in the top-level CMakeLists: the exported target name depends on how
# HDF5 was built, so it is not safe to write one in here.
target_link_libraries(bayests_hdf5_helper PUBLIC ${BAYESTS_HDF5_TARGET})
target_link_libraries(bayests_hdf5_helper PUBLIC HighFive)
target_link_libraries(bayests_hdf5_helper PUBLIC bayests_core)

# Link OpenMP for multi-threaded matrix operations
if(OpenMP_CXX_FOUND)
    target_link_libraries(bayests_hdf5_helper PUBLIC OpenMP::OpenMP_CXX)
endif()
