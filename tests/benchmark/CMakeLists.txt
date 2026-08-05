add_executable(clwrapper_benchmark main.cpp)
target_link_libraries(clwrapper_benchmark
  PRIVATE
    clwrapper
)
