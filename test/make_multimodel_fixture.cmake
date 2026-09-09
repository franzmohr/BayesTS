# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Franz X. Mohr
#
# Writes two models into one HDF5 file, each under its own group.
#
# A script rather than two add_test() commands, because the second call has to
# see the file the first one wrote: CTest gives no ordering between two commands
# of one test, and two tests chained with FIXTURES_SETUP would still race a
# -j run. Driven by bayests_add_multimodel_fixture() in CMakeLists.txt.
#
#   cmake -DGEN=<make_model_fixture> -DDEST=<file.h5>
#         -DMODEL=<name> -DGROUP_A=<group> -DGROUP_B=<group>
#         -P make_multimodel_fixture.cmake

foreach(_required GEN DEST MODEL GROUP_A GROUP_B)
    if(NOT DEFINED ${_required})
        message(FATAL_ERROR "make_multimodel_fixture.cmake needs -D${_required}")
    endif()
endforeach()

# The first call creates the file; the second is told to append so that it adds
# its model beside the first rather than starting over.
execute_process(
    COMMAND "${GEN}" "${DEST}" "${MODEL}" none 0 0 4 "${GROUP_A}"
    RESULT_VARIABLE _first)
if(NOT _first EQUAL 0)
    message(FATAL_ERROR "Writing ${MODEL} under ${GROUP_A} failed: ${_first}")
endif()

execute_process(
    COMMAND "${GEN}" "${DEST}" "${MODEL}" none 0 0 4 "${GROUP_B}" append
    RESULT_VARIABLE _second)
if(NOT _second EQUAL 0)
    message(FATAL_ERROR "Writing ${MODEL} under ${GROUP_B} failed: ${_second}")
endif()
