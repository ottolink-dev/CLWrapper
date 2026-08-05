#include "cl_wrapper/device_manager.hpp"
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
