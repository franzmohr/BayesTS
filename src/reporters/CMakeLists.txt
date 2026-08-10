

#####################################################################
# Reporters
#
# Implementations of bayests::Reporter for hosts that have a console to write
# to. Kept out of the core so that a host without one -- or with rules about
# who may write to it -- links its own instead.

add_library(bayests_console_reporter
            console_reporter.cpp console_reporter.h)

add_library(BayesTS::console_reporter ALIAS bayests_console_reporter)
target_link_libraries(bayests_console_reporter PRIVATE bayests_warnings)

target_link_libraries(bayests_console_reporter PUBLIC bayests_includes)
target_link_libraries(bayests_console_reporter PUBLIC bayests_core)
