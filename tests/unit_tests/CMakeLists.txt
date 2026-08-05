add_executable(clwrapper_unit_tests test_device_manager.cpp test_kernel_manager.cpp test_run.cpp)
target_link_libraries(clwrapper_unit_tests
  PRIVATE
    GTest::gtest
    GTest::gtest_main
    clwrapper
)

add_test(NAME clwrapper_unit_tests COMMAND clwrapper_unit_tests)
