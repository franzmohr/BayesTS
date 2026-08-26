
#####################################################################
# Regression harnesses
#
# Not installed and not part of a package: these exist to compare one build of
# the samplers against another. See golden_models.cpp for how the comparison is
# made reproducible.
#
# Target names carry the bayests_ prefix because CMake target names are global
# to a build tree, and these are created whenever BAYESTS_BUILD_TESTS is on --
# including when a parent project pulls BayesTS in with add_subdirectory().

add_executable(bayests_golden golden_models.cpp)

target_link_libraries(bayests_golden PRIVATE bayests_models)
target_link_libraries(bayests_golden PRIVATE bayests_hdf5_helper)

# Builds the inputs for the branches the recorded fixtures never reach.
add_executable(bayests_make_varsel_fixture make_varsel_fixture.cpp)

target_link_libraries(bayests_make_varsel_fixture PRIVATE bayests_hdf5_helper)

# Writes a model file from scratch for the samplers that have no recorded
# inputs. See make_model_fixture.cpp.
add_executable(bayests_make_model_fixture make_model_fixture.cpp)

target_link_libraries(bayests_make_model_fixture PRIVATE bayests_hdf5_helper)

target_link_libraries(bayests_golden PRIVATE bayests_warnings)
target_link_libraries(bayests_make_varsel_fixture PRIVATE bayests_warnings)
target_link_libraries(bayests_make_model_fixture PRIVATE bayests_warnings)

if(OpenMP_CXX_FOUND)
    target_link_libraries(bayests_golden PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(bayests_make_varsel_fixture PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(bayests_make_model_fixture PRIVATE OpenMP::OpenMP_CXX)
endif()

#####################################################################
# Unit tests
#
# A different kind of test from the harnesses above, and registered separately
# because it needs none of their machinery: no fixture, no recorded input, no
# pinned thread count. The transformation it covers is a change of basis rather
# than a draw, so it has an expected value that holds on every machine, and it
# asserts against it and returns non-zero -- which is what lets these be
# regressions rather than something to diff by hand.
#
# Linked against the algorithms library alone. That the test needs nothing from
# bayests_models is the point: an algorithm that can be tested without standing
# up a sampler belongs in src/core/algorithms.

add_executable(bayests_unit_vec_to_var unit_vec_to_var.cpp)
add_executable(bayests_unit_forecast_lags unit_forecast_lags.cpp)
add_executable(bayests_unit_stochvol unit_stochvol.cpp)
add_executable(bayests_unit_kalman unit_kalman.cpp)
add_executable(bayests_unit_chan_jeliazkov unit_chan_jeliazkov.cpp)

foreach(_unit bayests_unit_vec_to_var bayests_unit_forecast_lags bayests_unit_stochvol bayests_unit_kalman bayests_unit_chan_jeliazkov)
    target_link_libraries(${_unit} PRIVATE bayests_algorithms)
    target_link_libraries(${_unit} PRIVATE bayests_warnings)

    if(OpenMP_CXX_FOUND)
        target_link_libraries(${_unit} PRIVATE OpenMP::OpenMP_CXX)
    endif()
endforeach()

# The exceptions: these two reach into bayests_core rather than the algorithms
# library -- one covers validate(), the other runs two samplers against each
# other. Still no fixture: both build their inputs in memory.
#
# unit_vec_klgs_2010 is the odd one out in that it does draw, and so does depend
# on the RNG. It stays a unit test all the same, because what it asserts is a
# difference between two chains drawn from the same seed rather than the value of
# either -- an identity that holds on every machine, which is the line these are
# on the right side of.
add_executable(bayests_unit_identification unit_identification.cpp)
add_executable(bayests_unit_vec_klgs_2010 unit_vec_klgs_2010.cpp)

foreach(_core_unit bayests_unit_identification bayests_unit_vec_klgs_2010)
    target_link_libraries(${_core_unit} PRIVATE bayests_core)
    target_link_libraries(${_core_unit} PRIVATE bayests_warnings)

    if(OpenMP_CXX_FOUND)
        target_link_libraries(${_core_unit} PRIVATE OpenMP::OpenMP_CXX)
    endif()
endforeach()

#####################################################################
# Tests
#
# `ctest` is the entry point. Fixtures are generated into the build tree, never
# into the source tree: they are derived data, they are large -- a full set is
# hundreds of megabytes -- and a source tree that accumulates them cannot be
# reasoned about by size or cleaned by `git clean` without care.
#
#   cmake --build build/bin/<preset>
#   ctest --test-dir build/bin/<preset> --output-on-failure
#
# Each model file is produced by a `fixture.*` test and consumed by the
# `golden.*` test that follows it, wired through CTest's FIXTURES_SETUP /
# FIXTURES_REQUIRED so that running a single golden test by name regenerates
# just the input it needs.
#
# What these tests do and do not check: bayests_golden prints a fingerprint per
# posterior dataset and exits non-zero only if a fixture throws. So this is a
# smoke test over every sampler and code path, not a numerical regression test.
# Fingerprints are deliberately not checked in as expected values -- they shift
# in the last digits with the compiler, the BLAS and, when BAYESTS_NATIVE_ARCH
# is on, the CPU, so a committed baseline would fail on machines other than the
# one that recorded it. To use them as a regression check, record the output
# before a change and diff it against the output after, on the same machine and
# with the same build.

set(BAYESTS_FIXTURE_DIR "${CMAKE_CURRENT_BINARY_DIR}/fixtures")

# The samplers are only reproducible single-threaded, and bayests_golden
# refuses to run unless both are pinned.
set(_bayests_pin_threads "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")

# A test launched by ctest has to be able to find the shared libraries the
# samplers link -- Armadillo, HDF5 and the BLAS underneath them -- and neither
# platform manages that on its own when they come from a prefix that is not the
# system one. Prepend the same directories the install step resolves runtime
# dependencies from, so `ctest` works in a shell that has not been set up by hand.
#
# Windows has no RPATH at all, so the DLLs have to be on PATH. Unix has RPATH and
# usually needs no help, but not always: a library reached only as the transitive
# dependency of an imported target contributes no RPATH entry, so the link
# succeeds and the binary then dies at startup with "cannot open shared object
# file". A side-by-side prefix -- vcpkg, or a hand-built HDF5 -- is where this
# shows up.
#
# macOS is deliberately not covered: DYLD_LIBRARY_PATH is stripped from the
# environment of a protected process, so this mechanism cannot be relied on
# there, and nothing tests it.
set(_bayests_test_env_mod "")
if(WIN32)
    get_filename_component(_bayests_compiler_dir "${CMAKE_CXX_COMPILER}" DIRECTORY)
    set(_bayests_dll_dirs "${_bayests_compiler_dir}")
    foreach(_prefix IN LISTS CMAKE_PREFIX_PATH)
        list(APPEND _bayests_dll_dirs "${_prefix}/bin")
    endforeach()
    list(APPEND _bayests_dll_dirs ${BAYESTS_RUNTIME_DEP_DIRS})
    list(REMOVE_DUPLICATES _bayests_dll_dirs)

    foreach(_dir IN LISTS _bayests_dll_dirs)
        if(IS_DIRECTORY "${_dir}")
            list(APPEND _bayests_test_env_mod "PATH=path_list_prepend:${_dir}")
        endif()
    endforeach()
elseif(UNIX AND NOT APPLE)
    # Both trees, because a multi-configuration prefix keeps its debug build
    # under debug/lib and either may be the one that was linked.
    set(_bayests_so_dirs "")
    foreach(_prefix IN LISTS CMAKE_PREFIX_PATH)
        list(APPEND _bayests_so_dirs "${_prefix}/lib" "${_prefix}/debug/lib")
    endforeach()
    list(APPEND _bayests_so_dirs ${BAYESTS_RUNTIME_DEP_DIRS})
    list(REMOVE_DUPLICATES _bayests_so_dirs)

    foreach(_dir IN LISTS _bayests_so_dirs)
        if(IS_DIRECTORY "${_dir}")
            list(APPEND _bayests_test_env_mod "LD_LIBRARY_PATH=path_list_prepend:${_dir}")
        endif()
    endforeach()
endif()

# The unit tests, which need only the shared libraries on the path: no fixture
# to set up, and no thread pinning, since an exact identity does not care how
# many threads computed it.
foreach(_unit vec_to_var forecast_lags stochvol kalman chan_jeliazkov identification vec_klgs_2010)
    add_test(NAME "unit.${_unit}" COMMAND "bayests_unit_${_unit}")
    set_tests_properties("unit.${_unit}" PROPERTIES
        ENVIRONMENT_MODIFICATION "${_bayests_test_env_mod}")
endforeach()

# One generated model file plus the golden run over it.
#
#   bayests_make_model_fixture <dest.h5> <model> <none|ssvs|bvs> <covar 0|1>
#                              <structural 0|1> <h>
function(bayests_add_model_fixture name model varsel covar structural horizon)
    add_test(NAME "fixture.${name}"
        COMMAND bayests_make_model_fixture
                "${BAYESTS_FIXTURE_DIR}/${name}.h5"
                "${model}" "${varsel}" "${covar}" "${structural}" "${horizon}")
    set_tests_properties("fixture.${name}" PROPERTIES
        FIXTURES_SETUP "fx.${name}"
        ENVIRONMENT_MODIFICATION "${_bayests_test_env_mod}")

    add_test(NAME "golden.${name}"
        COMMAND bayests_golden "${BAYESTS_FIXTURE_DIR}/${name}.h5")
    set_tests_properties("golden.${name}" PROPERTIES
        FIXTURES_REQUIRED "fx.${name}"
        ENVIRONMENT "${_bayests_pin_threads}"
        ENVIRONMENT_MODIFICATION "${_bayests_test_env_mod}")
endfunction()

# The generation matrix. h=4 and k=3 throughout, so a forecast is 4x3 per draw.
#
# SSVS appears only for VarNormalGamma: the stochastic volatility and
# time-varying parameter samplers reject it (see the validate() overloads in
# src/core/inputs.cpp), so a fixture asking for it would be testing the error
# path, not the sampler.
#
# The `covar` flag is what decides whether a model has a psi block at all, and
# make_model_fixture only writes /model/priors/psi -- and with it psi's own
# varsel attribute -- when it is set. A selection scheme therefore only reaches
# the psi block in a fixture that asks for both, which is what the `-covar`
# suffixed selection rows below are for: without them the whole psi selection
# path went unexercised in every model, while `<scheme>` with covar=0 covered
# only the lag coefficients.
bayests_add_model_fixture(VarNormalGamma-plain      VarNormalGamma     none 0 0 4)
bayests_add_model_fixture(VarNormalGamma-ssvs       VarNormalGamma     ssvs 0 0 4)
bayests_add_model_fixture(VarNormalGamma-bvs        VarNormalGamma     bvs  0 0 4)
bayests_add_model_fixture(VarNormalGamma-covar      VarNormalGamma     none 1 0 4)
bayests_add_model_fixture(VarNormalGamma-structural VarNormalGamma     none 0 1 4)
bayests_add_model_fixture(VarNormalGamma-nofcst     VarNormalGamma     none 0 0 0)
bayests_add_model_fixture(VarNormalGamma-ssvs-covar VarNormalGamma     ssvs 1 0 4)
bayests_add_model_fixture(VarNormalGamma-bvs-covar  VarNormalGamma     bvs  1 0 4)

bayests_add_model_fixture(VarNormalStochvol-plain      VarNormalStochvol none 0 0 4)
bayests_add_model_fixture(VarNormalStochvol-bvs        VarNormalStochvol bvs  0 0 4)
bayests_add_model_fixture(VarNormalStochvol-covar      VarNormalStochvol none 1 0 4)
bayests_add_model_fixture(VarNormalStochvol-structural VarNormalStochvol none 0 1 4)
bayests_add_model_fixture(VarNormalStochvol-nofcst     VarNormalStochvol none 0 0 0)
bayests_add_model_fixture(VarNormalStochvol-bvs-covar  VarNormalStochvol bvs  1 0 4)

bayests_add_model_fixture(VarTvpGamma-plain      VarTvpGamma none 0 0 4)
bayests_add_model_fixture(VarTvpGamma-bvs        VarTvpGamma bvs  0 0 4)
bayests_add_model_fixture(VarTvpGamma-covar      VarTvpGamma none 1 0 4)
bayests_add_model_fixture(VarTvpGamma-structural VarTvpGamma none 0 1 4)
bayests_add_model_fixture(VarTvpGamma-nofcst     VarTvpGamma none 0 0 0)
bayests_add_model_fixture(VarTvpGamma-bvs-covar  VarTvpGamma bvs  1 0 4)

bayests_add_model_fixture(VarTvpWishart-plain      VarTvpWishart none 0 0 4)
bayests_add_model_fixture(VarTvpWishart-bvs        VarTvpWishart bvs  0 0 4)
bayests_add_model_fixture(VarTvpWishart-nofcst     VarTvpWishart none 0 0 0)

bayests_add_model_fixture(VarTvpStochvol-plain      VarTvpStochvol none 0 0 4)
bayests_add_model_fixture(VarTvpStochvol-bvs        VarTvpStochvol bvs  0 0 4)
bayests_add_model_fixture(VarTvpStochvol-covar      VarTvpStochvol none 1 0 4)
bayests_add_model_fixture(VarTvpStochvol-structural VarTvpStochvol none 0 1 4)
bayests_add_model_fixture(VarTvpStochvol-nofcst     VarTvpStochvol none 0 0 0)
bayests_add_model_fixture(VarTvpStochvol-bvs-covar  VarTvpStochvol bvs  1 0 4)

# The VECs that have a generator -- every one but VecNormalWishart. Their
# dimensions differ from the rows above (rank 1, a restricted constant and one
# lagged difference) and their forecast regressors are in levels; see the VEC
# block in make_model_fixture.cpp. SSVS reaches VecNormalGamma only, for the same
# reason it reaches VarNormalGamma only.
#
# The structural rows are what cover the level-VAR forecast's contemporaneous
# split from the VEC side: every VEC forecast converts to a level VAR and hands
# the path to VarNormalWishartSampler, so these five and the VAR rows above
# exercise the same code from both ends.
# No selection rows: VecKlgs2010 draws the coefficients without forming the SUR
# design matrix a scheme would act on, and validate() refuses either one. No
# structural row either, for the reason the Wishart models below have none.
bayests_add_model_fixture(VecKlgs2010-plain  VecKlgs2010 none 0 0 4)
bayests_add_model_fixture(VecKlgs2010-nofcst VecKlgs2010 none 0 0 0)

bayests_add_model_fixture(VecNormalGamma-plain      VecNormalGamma none 0 0 4)
bayests_add_model_fixture(VecNormalGamma-ssvs       VecNormalGamma ssvs 0 0 4)
bayests_add_model_fixture(VecNormalGamma-bvs        VecNormalGamma bvs  0 0 4)
bayests_add_model_fixture(VecNormalGamma-covar      VecNormalGamma none 1 0 4)
bayests_add_model_fixture(VecNormalGamma-structural VecNormalGamma none 0 1 4)
bayests_add_model_fixture(VecNormalGamma-nofcst     VecNormalGamma none 0 0 0)
bayests_add_model_fixture(VecNormalGamma-bvs-covar  VecNormalGamma bvs  1 0 4)

bayests_add_model_fixture(VecNormalStochvol-plain     VecNormalStochvol none 0 0 4)
bayests_add_model_fixture(VecNormalStochvol-bvs       VecNormalStochvol bvs  0 0 4)
bayests_add_model_fixture(VecNormalStochvol-covar     VecNormalStochvol none 1 0 4)
bayests_add_model_fixture(VecNormalStochvol-structural VecNormalStochvol none 0 1 4)
bayests_add_model_fixture(VecNormalStochvol-nofcst    VecNormalStochvol none 0 0 0)
bayests_add_model_fixture(VecNormalStochvol-bvs-covar VecNormalStochvol bvs  1 0 4)

# No covar row: like VarTvpWishart, this model carries no psi block.
bayests_add_model_fixture(VecTvpWishart-plain  VecTvpWishart none 0 0 4)
bayests_add_model_fixture(VecTvpWishart-bvs    VecTvpWishart bvs  0 0 4)
bayests_add_model_fixture(VecTvpWishart-nofcst VecTvpWishart none 0 0 0)

bayests_add_model_fixture(VecTvpGamma-plain     VecTvpGamma none 0 0 4)
bayests_add_model_fixture(VecTvpGamma-bvs       VecTvpGamma bvs  0 0 4)
bayests_add_model_fixture(VecTvpGamma-covar     VecTvpGamma none 1 0 4)
bayests_add_model_fixture(VecTvpGamma-structural VecTvpGamma none 0 1 4)
bayests_add_model_fixture(VecTvpGamma-nofcst    VecTvpGamma none 0 0 0)
bayests_add_model_fixture(VecTvpGamma-bvs-covar VecTvpGamma bvs  1 0 4)

bayests_add_model_fixture(VecTvpStochvol-plain     VecTvpStochvol none 0 0 4)
bayests_add_model_fixture(VecTvpStochvol-bvs       VecTvpStochvol bvs  0 0 4)
bayests_add_model_fixture(VecTvpStochvol-covar     VecTvpStochvol none 1 0 4)
bayests_add_model_fixture(VecTvpStochvol-structural VecTvpStochvol none 0 1 4)
bayests_add_model_fixture(VecTvpStochvol-nofcst    VecTvpStochvol none 0 0 0)
bayests_add_model_fixture(VecTvpStochvol-bvs-covar VecTvpStochvol bvs  1 0 4)

#####################################################################
# VarNormalWishart
#
# make_model_fixture cannot write this model, so its fixtures are derived from a
# recorded file with make_varsel_fixture. No such file is in the repository --
# they are hundreds of megabytes of sampler output -- so these tests exist only
# when one is supplied:
#
#   cmake --preset <preset> -DBAYESTS_WISHART_FIXTURE=path/to/recorded.h5
#
# and are skipped, with a note at configure time, when it is not.
set(BAYESTS_WISHART_FIXTURE "" CACHE FILEPATH
    "Recorded VarNormalWishart model file the w-* regression fixtures are derived from")

if(BAYESTS_WISHART_FIXTURE AND EXISTS "${BAYESTS_WISHART_FIXTURE}")
    foreach(_scheme none ssvs bvs)
        add_test(NAME "fixture.w-${_scheme}"
            COMMAND bayests_make_varsel_fixture
                    "${BAYESTS_WISHART_FIXTURE}"
                    "${BAYESTS_FIXTURE_DIR}/w-${_scheme}.h5"
                    "${_scheme}" 4)
        set_tests_properties("fixture.w-${_scheme}" PROPERTIES
            FIXTURES_SETUP "fx.w-${_scheme}"
            ENVIRONMENT_MODIFICATION "${_bayests_test_env_mod}")

        add_test(NAME "golden.w-${_scheme}"
            COMMAND bayests_golden "${BAYESTS_FIXTURE_DIR}/w-${_scheme}.h5")
        set_tests_properties("golden.w-${_scheme}" PROPERTIES
            FIXTURES_REQUIRED "fx.w-${_scheme}"
            ENVIRONMENT "${_bayests_pin_threads}"
            ENVIRONMENT_MODIFICATION "${_bayests_test_env_mod}")
    endforeach()
    message(STATUS "VarNormalWishart fixtures derived from ${BAYESTS_WISHART_FIXTURE}")
else()
    message(STATUS "BAYESTS_WISHART_FIXTURE not set - VarNormalWishart tests skipped")
endif()

#####################################################################
# VecNormalWishart
#
# Same arrangement as VarNormalWishart above and for the same reason -- no tool
# here can write a VEC model file, so the test exists only when one is supplied:
#
#   cmake --preset <preset> -DBAYESTS_VEC_FIXTURE=path/to/recorded.h5
#
# No fixture step, unlike the block above. That one derives three files from a
# single recording, one per selection scheme, and so needs a generator; this
# needs the file exactly as it stands. Nothing has to be copied first either:
# bayests_golden stages its own copy under the temporary directory and clears the
# outputs there, so the supplied file is read and never written -- see
# stage_fixture() in golden_models.cpp.
set(BAYESTS_VEC_FIXTURE "" CACHE FILEPATH
    "Recorded VecNormalWishart model file the vec-* regression test reads")

if(BAYESTS_VEC_FIXTURE AND EXISTS "${BAYESTS_VEC_FIXTURE}")
    add_test(NAME "golden.vec-none" COMMAND bayests_golden "${BAYESTS_VEC_FIXTURE}")
    set_tests_properties("golden.vec-none" PROPERTIES
        ENVIRONMENT "${_bayests_pin_threads}"
        ENVIRONMENT_MODIFICATION "${_bayests_test_env_mod}")
    message(STATUS "VecNormalWishart test reads ${BAYESTS_VEC_FIXTURE}")
else()
    message(STATUS "BAYESTS_VEC_FIXTURE not set - VecNormalWishart tests skipped")
endif()

# VarTvpWishart carries no psi block, so it has no covar fixture -- and no
# structural one either: its Wishart prior leaves Sigma unrestricted, which
# validate() now refuses alongside contemporaneous coefficients. The same goes
# for VecTvpWishart above. Every remaining structural row pairs A_0 with a
# diagonal covariance, which is the only combination that identifies it. It is also the
# one model whose R-side implementation no longer exists, which makes the
# fingerprints recorded from these fixtures the only baseline it has: record
# them before a change and diff them after, on the same machine and build.
message(STATUS "Regression tests registered - run with: ctest --test-dir <binaryDir>")
