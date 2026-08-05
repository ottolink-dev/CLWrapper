/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */

/**
 * @file device_manager.hpp
 * @author Otto Link (otto.link.bv@gmail.com)
 * @brief
 *
 * @copyright Copyright (c) 2025
 */
#pragma once
#include <map>

#include <CL/opencl.hpp>

namespace clwrapper
{

class DeviceManager
{
public:
  // Get the singleton instance
  static DeviceManager &get_instance()
  {
    static DeviceManager instance;
    return instance;
  }

  static bool is_ready();

  // Get the OpenCL device attached to the singleton instance
  static cl::Device device()
  {
    return DeviceManager::get_instance().get_device();
  }

  // list all the available devices
  std::map<size_t, std::string> get_available_devices();

  // Access the OpenCL device
  cl::Device get_device() const;

  size_t get_device_id() const
  {
    return this->device_id;
  }

  bool set_device(size_t platform_id);

  void set_device_type(cl_device_type new_device_type)
  {
    this->device_type = new_device_type;
  }

private:
  cl::Device cl_device;

  size_t device_id = 0;

  // allowed device type (CL_DEVICE_TYPE_ALL | GPU | CPU)
  cl_device_type device_type = CL_DEVICE_TYPE_ALL;

  // Private constructor
  DeviceManager();

  // Delete copy constructor and assignment operator to enforce singleton
  DeviceManager(const DeviceManager &) = delete;
  DeviceManager &operator=(const DeviceManager &) = delete;
};

void log_device_infos(cl::Device cl_device);

} // namespace clwrapper
