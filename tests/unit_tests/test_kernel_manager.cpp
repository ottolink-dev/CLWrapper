#include "cl_wrapper/kernel_manager.hpp"
#include <gtest/gtest.h>

using namespace clwrapper;

TEST(KernelManagerTest, SingletonInstanceTest)
{
  KernelManager &km = KernelManager::get_instance();
  EXPECT_NO_THROW({
    cl::Context ctx = km.get_context();
    cl::Context static_ctx = KernelManager::context();
  });

  // Test build options
  km.set_build_options("-cl-mad-enable");
  EXPECT_EQ(km.get_build_options(), "-cl-mad-enable");

  // Restore empty build options
  km.set_build_options("");
}

TEST(KernelManagerTest, AddKernelTest)
{
  KernelManager &km = KernelManager::get_instance();

  // Clear sources
  km.clear_sources();
  EXPECT_TRUE(km.get_full_sources().empty());

  const std::string kernel_code =
      "__kernel void simple_add(__global const float* a, __global const float* "
      "b, __global float* c) {\n"
      "    int id = get_global_id(0);\n"
      "    c[id] = a[id] + b[id];\n"
      "}\n";

  // Add kernel and verify it gets built
  km.add_kernel(kernel_code, true);
  EXPECT_EQ(km.get_full_sources(), kernel_code);
  EXPECT_TRUE(km.get_program().get() != nullptr);
}

TEST(KernelManagerTest, SourceStackingTest)
{
  KernelManager &km = KernelManager::get_instance();
  km.clear_sources();

  const std::string part1 = "__kernel void add_kernel(__global const float* a, "
                            "__global const float* b, __global float* c) {\n"
                            "    int id = get_global_id(0);\n"
                            "    c[id] = a[id] + b[id];\n"
                            "}\n";

  const std::string part2 = "__kernel void sub_kernel(__global const float* a, "
                            "__global const float* b, __global float* c) {\n"
                            "    int id = get_global_id(0);\n"
                            "    c[id] = a[id] - b[id];\n"
                            "}\n";

  // Add part1 but do not build yet
  km.add_kernel(part1, true, false);
  EXPECT_EQ(km.get_full_sources(), part1);

  // Add part2 and do not build yet
  km.add_kernel(part2, false, false);
  EXPECT_EQ(km.get_full_sources(), part1 + part2);

  // Now trigger build manually
  EXPECT_NO_THROW(km.build_program());
  EXPECT_TRUE(km.get_program().get() != nullptr);
}
