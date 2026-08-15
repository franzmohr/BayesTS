

#####################################################################
# Library
#
# The BaseModel classes the command line drives. All six are now front-ends:
# each decides what still needs computing for a given file, hands the work to
# bayests_core and writes the results back. No numerics are left here.

add_library(bayests_models
            models.h
            model_factory.cpp
            VarNormalGamma.cpp
            VarNormalStochvol.cpp
            VarNormalWishart.cpp
            VarTvpGamma.cpp
            VarTvpWishart.cpp
            VarTvpStochvol.cpp
            VecNormalWishart.cpp)

add_library(BayesTS::models ALIAS bayests_models)
target_link_libraries(bayests_models PRIVATE bayests_warnings)

#####################################################################
# Include directories
target_link_libraries(bayests_models PUBLIC bayests_includes)

#####################################################################
# Linking
target_link_libraries(bayests_models PUBLIC armadillo)
# See src/io/hdf5/CMakeLists.txt: the name is resolved once at the top level.
target_link_libraries(bayests_models PUBLIC ${BAYESTS_HDF5_TARGET})
target_link_libraries(bayests_models PUBLIC bayests_hdf5_helper)
target_link_libraries(bayests_models PUBLIC bayests_core)
target_link_libraries(bayests_models PRIVATE bayests_console_reporter)

# Link OpenMP for multi-threaded matrix operations
if(OpenMP_CXX_FOUND)
    target_link_libraries(bayests_models PUBLIC OpenMP::OpenMP_CXX)
endif()
