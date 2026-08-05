#include <gtest/gtest.h>
#include "cl_wrapper/kernel_manager.hpp"

using namespace clwrapper;

TEST(KernelManagerTest, SingletonInstanceTest) {
    KernelManager& km = KernelManager::get_instance();
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

TEST(KernelManagerTest, AddKernelTest) {
    KernelManager& km = KernelManager::get_instance();
    
    // Clear sources
    km.clear_sources();
    EXPECT_TRUE(km.get_full_sources().empty());
    
    const std::string kernel_code = 
        "__kernel void simple_add(__global const float* a, __global const float* b, __global float* c) {\n"
        "    int id = get_global_id(0);\n"
        "    c[id] = a[id] + b[id];\n"
        "}\n";
        
    // Add kernel and verify it gets built
    km.add_kernel(kernel_code, true);
    EXPECT_EQ(km.get_full_sources(), kernel_code);
    EXPECT_TRUE(km.get_program().get() != nullptr);
}
