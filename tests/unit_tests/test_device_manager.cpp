#include "cl_wrapper/device_manager.hpp"
#include "cl_wrapper/kernel_manager.hpp"
#include "cl_wrapper/run.hpp"
#include <gtest/gtest.h>

using namespace clwrapper;

TEST(DeviceManagerTest, IsReadyTest)
{
  EXPECT_TRUE(DeviceManager::is_ready());
}

TEST(DeviceManagerTest, GetDeviceTest)
{
  cl::Device dev = DeviceManager::device();
  EXPECT_TRUE(dev.get() != nullptr);
}

TEST(DeviceManagerTest, AvailableDevicesTest)
{
  auto devices = DeviceManager::get_instance().get_available_devices();
  EXPECT_FALSE(devices.empty());

  auto all_devices = DeviceManager::get_instance().get_all_available_devices();
  EXPECT_FALSE(all_devices.empty());

  // Verify that the size of all_devices is at least the size of devices
  EXPECT_GE(all_devices.size(), devices.size());
}

TEST(DeviceManagerTest, DeviceMetadataTest)
{
  DeviceManager &dm = DeviceManager::get_instance();

  std::string    name = dm.get_device_name();
  std::string    vendor = dm.get_device_vendor();
  std::string    version = dm.get_device_version();
  cl_device_type type = dm.get_device_type();

  EXPECT_FALSE(name.empty());
  EXPECT_FALSE(vendor.empty());
  EXPECT_FALSE(version.empty());
  EXPECT_TRUE(type & (CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_CPU |
                      CL_DEVICE_TYPE_ACCELERATOR | CL_DEVICE_TYPE_DEFAULT));

  EXPECT_GT(dm.get_max_compute_units(), 0);
  EXPECT_GT(dm.get_max_work_group_size(), 0);
  EXPECT_GT(dm.get_global_mem_size(), 0);
  EXPECT_GT(dm.get_local_mem_size(), 0);
}

TEST(DeviceManagerTest, PlatformMetadataTest)
{
  DeviceManager &dm = DeviceManager::get_instance();

  std::string plat_name = dm.get_platform_name();
  std::string plat_vendor = dm.get_platform_vendor();
  std::string plat_version = dm.get_platform_version();

  EXPECT_FALSE(plat_name.empty());
  EXPECT_FALSE(plat_vendor.empty());
  EXPECT_FALSE(plat_version.empty());
}

TEST(DeviceManagerTest, SetDeviceBoundsCheckingTest)
{
  DeviceManager &dm = DeviceManager::get_instance();

  // Test set_device with an invalid platform ID (extremely large)
  EXPECT_FALSE(dm.set_device(999999));

  // Test set_device with an invalid device index on platform 0
  EXPECT_FALSE(dm.set_device(0, 999999));

  // Test set_device with a valid platform (current active one)
  size_t current_platform = dm.get_platform_id();
  size_t current_device = dm.get_device_index();
  EXPECT_TRUE(dm.set_device(current_platform, current_device));
}

TEST(DeviceManagerTest, SetDeviceTypeFilterAutoRediscoveryTest)
{
  DeviceManager &dm = DeviceManager::get_instance();

  // Remember current selection
  size_t current_platform = dm.get_platform_id();
  size_t current_device = dm.get_device_index();

  // Try to set device type filter to CPU
  dm.set_device_type(CL_DEVICE_TYPE_CPU);

  // Verify that the newly selected device is a CPU
  EXPECT_TRUE(dm.get_device_type() & CL_DEVICE_TYPE_CPU);

  // Restore original configuration
  dm.set_device_type(CL_DEVICE_TYPE_ALL);
  dm.set_device(current_platform, current_device);
}

TEST(DeviceManagerTest, DeviceSwitchAndExecutionTest)
{
  DeviceManager &dm = DeviceManager::get_instance();
  KernelManager &km = KernelManager::get_instance();

  size_t original_platform = dm.get_platform_id();
  size_t original_device = dm.get_device_index();

  auto devices = dm.get_all_available_devices();

  const std::string code =
      "__kernel void simple_add(__global const float* a, __global const float* "
      "b, __global float* c) {\n"
      "    int id = get_global_id(0);\n"
      "    c[id] = a[id] + b[id];\n"
      "}\n";

  for (auto &[coords, name] : devices)
  {
    // Select device
    ASSERT_TRUE(dm.set_device(coords.first, coords.second));

    // Build program for this device
    km.add_kernel(code, true, true);

    // Execute a run
    clwrapper::Run     run("simple_add");
    int                n = 5;
    std::vector<float> a(n, 1.0f);
    std::vector<float> b(n, 2.0f);
    std::vector<float> c(n, 0.0f);

    run.bind_buffer<float>("a", a);
    run.bind_buffer<float>("b", b);
    run.bind_buffer<float>("c", c);

    run.write_buffer("a");
    run.write_buffer("b");

    EXPECT_NO_THROW(run.execute(n));

    run.read_buffer("c");

    for (int i = 0; i < n; ++i)
    {
      EXPECT_FLOAT_EQ(c[i], 3.0f);
    }
  }

  // Restore original device
  dm.set_device(original_platform, original_device);
}
