# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Franz X. Mohr

# That `bayests check` refuses a model file a run would refuse, with exit code 1
# and the run's own wording.
#
#   cmake -DBAYESTS=<bayests> -DFIXTURE=<model.h5> -DPATTERN=<regex> -P check_refuses.cmake
#
# CTest can test either the exit code (WILL_FAIL) or the output
# (PASS_REGULAR_EXPRESSION, which ignores the exit code) but not both, and a
# check that refused for some other reason -- an unreadable file, a reader that
# moved -- would pass either one alone.

foreach(_required BAYESTS FIXTURE PATTERN)
    if(NOT DEFINED ${_required})
        message(FATAL_ERROR "check_refuses.cmake needs -D${_required}=...")
    endif()
endforeach()

execute_process(
    COMMAND "${BAYESTS}" check "${FIXTURE}"
    RESULT_VARIABLE _code
    OUTPUT_VARIABLE _out
    ERROR_VARIABLE _err)

if(NOT _code STREQUAL "1")
    message(FATAL_ERROR "expected exit 1 from bayests check, got ${_code}\n${_out}\n${_err}")
endif()
if(NOT _err MATCHES "${PATTERN}")
    message(FATAL_ERROR "bayests check refused, but not with '${PATTERN}'\n${_out}\n${_err}")
endif()
message(STATUS "ok: refused with '${PATTERN}'")
