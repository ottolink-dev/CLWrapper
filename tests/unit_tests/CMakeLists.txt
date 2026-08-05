add_executable(unit_tests test_device_manager.cpp test_kernel_manager.cpp test_run.cpp)
target_link_libraries(unit_tests
  PRIVATE
    GTest::gtest
    GTest::gtest_main
    clwrapper
)

add_test(NAME unit_tests COMMAND unit_tests)
