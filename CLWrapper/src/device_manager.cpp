/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <algorithm>
#include <mutex>
#include <shared_mutex>

#include "cl_wrapper/device_manager.hpp"
#include "cl_wrapper/logger.hpp"

namespace clwrapper
{

DeviceManager::DeviceManager()
{
  Logger::log()->trace("DeviceManager::DeviceManager");
  this->select_optimal_device();
}

cl::Device DeviceManager::device()
{
  return DeviceManager::get_instance().get_device();
}

double DeviceManager::evaluate_device(const cl::Device &device) const
{
  double score = 0.0;

  try
  {
    // 1. Device Type Priority
    cl_device_type type = device.getInfo<CL_DEVICE_TYPE>();
    if (type & CL_DEVICE_TYPE_GPU)
    {
      score += 10000.0;

      // 2. Discrete vs Integrated GPU (favor discrete)
      cl_bool unified_mem = CL_TRUE;
      try
      {
        unified_mem = device.getInfo<CL_DEVICE_HOST_UNIFIED_MEMORY>();
      }
      catch (...)
      {
      } // Fallback if unified memory attribute is not supported

      if (unified_mem == CL_FALSE)
      {
        score += 5000.0;
      }
    }
    else if (type & CL_DEVICE_TYPE_ACCELERATOR)
    {
      score += 5000.0;
    }
    else if (type & CL_DEVICE_TYPE_CPU)
    {
      score += 1000.0;
    }
    else
    {
      score += 100.0;
    }

    // 3. Compute capacity (Compute Units * Clock Frequency)
    cl_uint compute_units = 1;
    cl_uint clock_freq = 1;
    try
    {
      compute_units = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
      clock_freq = device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>();
    }
    catch (...)
    {
    }

    score += static_cast<double>(compute_units) * clock_freq * 1e-3;

    // 4. Memory capacity tie-breaker
    cl_ulong global_mem = 0;
    try
    {
      global_mem = device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
    }
    catch (...)
    {
    }

    score += static_cast<double>(global_mem) * 1e-12;
  }
  catch (...)
  {
  }

  return score;
}

std::map<std::pair<size_t, size_t>, std::string> DeviceManager::
    get_all_available_devices()
{
  std::shared_lock<std::shared_mutex>              lock(this->state_mutex);
  std::map<std::pair<size_t, size_t>, std::string> device_map = {};

  try
  {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    if (platforms.empty()) return device_map;

    for (size_t kp = 0; kp < platforms.size(); kp++)
    {
      std::vector<cl::Device> devices;
      platforms[kp].getDevices(this->device_type, &devices);

      for (size_t kd = 0; kd < devices.size(); kd++)
      {
        std::string name = platforms[kp].getInfo<CL_PLATFORM_VENDOR>() + "/" +
                           platforms[kp].getInfo<CL_PLATFORM_NAME>() + "/" +
                           devices[kd].getInfo<CL_DEVICE_NAME>();

        device_map[{kp, kd}] = name;
      }
    }
  }
  catch (...)
  {
  }

  return device_map;
}

std::map<size_t, std::string> DeviceManager::get_available_devices()
{
  std::shared_lock<std::shared_mutex> lock(this->state_mutex);
  std::map<size_t, std::string>       device_map = {};

  try
  {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    if (platforms.empty()) return device_map;

    for (size_t kp = 0; kp < platforms.size(); kp++)
    {
      std::vector<cl::Device> devices;
      platforms[kp].getDevices(this->device_type, &devices);

      if (!devices.empty())
      {
        std::string name = platforms[kp].getInfo<CL_PLATFORM_VENDOR>() + "/" +
                           platforms[kp].getInfo<CL_PLATFORM_NAME>() + "/" +
                           devices[0].getInfo<CL_DEVICE_NAME>();

        device_map[kp] = name;
      }
    }
  }
  catch (...)
  {
  }

  return device_map;
}

cl::Device DeviceManager::get_device() const
{
  std::shared_lock<std::shared_mutex> lock(this->state_mutex);
  return this->cl_device;
}

size_t DeviceManager::get_device_id() const
{
  std::shared_lock<std::shared_mutex> lock(this->state_mutex);
  return this->platform_id;
}

size_t DeviceManager::get_device_index() const
{
  std::shared_lock<std::shared_mutex> lock(this->state_mutex);
  return this->device_index;
}

std::string DeviceManager::get_device_name() const
{
  std::shared_lock<std::shared_mutex> lock(this->state_mutex);
  try
  {
    return this->cl_device.getInfo<CL_DEVICE_NAME>();
  }
  catch (...)
  {
    return "";
  }
}

cl_device_type DeviceManager::get_device_type() const
{
  std::shared_lock<std::shared_mutex> lock(this->state_mutex);
  try
  {
    return this->cl_device.getInfo<CL_DEVICE_TYPE>();
  }
  catch (...)
  {
    return CL_DEVICE_TYPE_DEFAULT;
  }
}

std::string DeviceManager::get_device_vendor() const
{
  std::shared_lock<std::shared_mutex> lock(this->state_mutex);
  try
  {
    return this->cl_device.getInfo<CL_DEVICE_VENDOR>();
  }
  catch (...)
  {
    return "";
  }
}

std::string DeviceManager::get_device_version() const
{
  std::shared_lock<std::shared_mutex> lock(this->state_mutex);
  try
  {
    return this->cl_device.getInfo<CL_DEVICE_VERSION>();
  }
  catch (...)
  {
    return "";
  }
}

cl_ulong DeviceManager::get_global_mem_size() const
{
  std::shared_lock<std::shared_mutex> lock(this->state_mutex);
  try
  {
    return this->cl_device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
  }
  catch (...)
  {
    return 0;
  }
}

DeviceManager &DeviceManager::get_instance()
{
  static DeviceManager instance;
  return instance;
}

cl_ulong DeviceManager::get_local_mem_size() const
{
  std::shared_lock<std::shared_mutex> lock(this->state_mutex);
  try
  {
    return this->cl_device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
  }
  catch (...)
  {
    return 0;
  }
}

cl_uint DeviceManager::get_max_compute_units() const
{
  std::shared_lock<std::shared_mutex> lock(this->state_mutex);
  try
  {
    return this->cl_device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
  }
  catch (...)
  {
    return 0;
  }
}

size_t DeviceManager::get_max_work_group_size() const
{
  std::shared_lock<std::shared_mutex> lock(this->state_mutex);
  try
  {
    return this->cl_device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
  }
  catch (...)
  {
    return 0;
  }
}

size_t DeviceManager::get_platform_id() const
{
  std::shared_lock<std::shared_mutex> lock(this->state_mutex);
  return this->platform_id;
}

std::string DeviceManager::get_platform_name() const
{
  std::shared_lock<std::shared_mutex> lock(this->state_mutex);
  try
  {
    cl::Platform platform = this->cl_device.getInfo<CL_DEVICE_PLATFORM>();
    return platform.getInfo<CL_PLATFORM_NAME>();
  }
  catch (...)
  {
    return "";
  }
}

std::string DeviceManager::get_platform_vendor() const
{
  std::shared_lock<std::shared_mutex> lock(this->state_mutex);
  try
  {
    cl::Platform platform = this->cl_device.getInfo<CL_DEVICE_PLATFORM>();
    return platform.getInfo<CL_PLATFORM_VENDOR>();
  }
  catch (...)
  {
    return "";
  }
}

std::string DeviceManager::get_platform_version() const
{
  std::shared_lock<std::shared_mutex> lock(this->state_mutex);
  try
  {
    cl::Platform platform = this->cl_device.getInfo<CL_DEVICE_PLATFORM>();
    return platform.getInfo<CL_PLATFORM_VERSION>();
  }
  catch (...)
  {
    return "";
  }
}

bool DeviceManager::is_ready()
{
  try
  {
    DeviceManager::get_instance();
  }
  catch (const std::exception &e)
  {
    Logger::log()->error("Error: {}", e.what());
    return false;
  }
  catch (...)
  {
    Logger::log()->error("Unknown error");
    return false;
  }

  return true;
}

void DeviceManager::select_optimal_device()
{
  // initialize the device (example: first GPU)
  Logger::log()->trace("initializing OpenCL devices...");

  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);

  if (platforms.empty()) throw std::runtime_error("No OpenCL platforms found!");

  int    best_platform_idx = -1;
  int    best_device_idx = -1;
  double best_score = -1.0;

  Logger::log()->trace("checking device performances...");

  for (size_t kp = 0; kp < platforms.size(); kp++)
  {
    Logger::log()->trace("checking platform: {} - {}",
                         platforms[kp].getInfo<CL_PLATFORM_VENDOR>().c_str(),
                         platforms[kp].getInfo<CL_PLATFORM_NAME>().c_str());

    std::vector<cl::Device> devices;
    platforms[kp].getDevices(this->device_type, &devices);

    for (size_t kd = 0; kd < devices.size(); kd++)
    {
      double score = this->evaluate_device(devices[kd]);
      Logger::log()->trace("rating - device: {}, score: {}",
                           devices[kd].getInfo<CL_DEVICE_NAME>().c_str(),
                           score);

      if (score > best_score)
      {
        best_score = score;
        best_platform_idx = static_cast<int>(kp);
        best_device_idx = static_cast<int>(kd);
      }
    }
  }

  if (best_platform_idx == -1 || best_device_idx == -1)
  {
    throw std::runtime_error(
        "No OpenCL devices matching the device type filter were found!");
  }

  // eventually assign the platform / device
  std::vector<cl::Device> devices;
  platforms[best_platform_idx].getDevices(this->device_type, &devices);
  if (devices.empty())
  {
    throw std::runtime_error(
        "Selected platform has no devices matching the device type filter!");
  }
  this->cl_device = devices[best_device_idx];
  this->platform_id = best_platform_idx;
  this->device_index = best_device_idx;

  Logger::log()->info("Selected OpenCL device: {}",
                      this->cl_device.getInfo<CL_DEVICE_NAME>().c_str());

  log_device_infos(this->cl_device);
}

bool DeviceManager::set_device(size_t platform_id)
{
  return this->set_device(platform_id, 0);
}

bool DeviceManager::set_device(size_t platform_id, size_t device_index)
{
  std::unique_lock<std::shared_mutex> lock(this->state_mutex);
  std::vector<cl::Platform>           platforms;
  cl::Platform::get(&platforms);

  if (platforms.empty()) throw std::runtime_error("No OpenCL platforms found!");

  if (platform_id >= platforms.size())
  {
    Logger::log()->error(
        "Platform ID {} is out of bounds (total platforms: {})",
        platform_id,
        platforms.size());
    return false;
  }

  std::vector<cl::Device> devices;
  platforms[platform_id].getDevices(this->device_type, &devices);

  if (device_index >= devices.size())
  {
    Logger::log()->error(
        "Device index {} is out of bounds for platform {} (total devices: {})",
        device_index,
        platform_id,
        devices.size());
    return false;
  }

  this->cl_device = devices[device_index];
  this->platform_id = platform_id;
  this->device_index = device_index;

  Logger::log()->trace("OpenCL device: {}",
                       this->cl_device.getInfo<CL_DEVICE_NAME>().c_str());
  return true;
}

void DeviceManager::set_device_type(cl_device_type new_device_type)
{
  std::unique_lock<std::shared_mutex> lock(this->state_mutex);
  this->device_type = new_device_type;
  this->select_optimal_device();
}

void log_device_infos(cl::Device cl_device)
{
  try
  {
    Logger::log()->info("- device Name: {}",
                        cl_device.getInfo<CL_DEVICE_NAME>().c_str());
    Logger::log()->info(" - device Vendor: {}",
                        cl_device.getInfo<CL_DEVICE_VENDOR>().c_str());
    Logger::log()->info(" - device Version: {}",
                        cl_device.getInfo<CL_DEVICE_VERSION>().c_str());

    switch (cl_device.getInfo<CL_DEVICE_TYPE>())
    {
    case CL_DEVICE_TYPE_GPU: Logger::log()->info(" - device Type: GPU"); break;
    case CL_DEVICE_TYPE_CPU: Logger::log()->info(" - device Type: CPU"); break;
    case CL_DEVICE_TYPE_ACCELERATOR:
      Logger::log()->info(" - device Type: ACCELERATOR");
      break;
    default: Logger::log()->info(" - device Type: unknown");
    }
  }
  catch (...)
  {
  }
}

} // namespace clwrapper
