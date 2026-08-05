/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <algorithm>

#include "cl_wrapper/device_manager.hpp"
#include "cl_wrapper/logger.hpp"

namespace clwrapper
{

namespace
{
bool helper_find_string_insensitive(const std::string &text,
                                    const std::string &word)
{
  // https://stackoverflow.com/questions/3152241
  auto it = std::search(text.begin(),
                        text.end(),
                        word.begin(),
                        word.end(),
                        [](unsigned char ch1, unsigned char ch2)
                        { return std::toupper(ch1) == std::toupper(ch2); });
  return (it != text.end());
}
} // namespace

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

DeviceManager::DeviceManager()
{
  Logger::log()->trace("DeviceManager::DeviceManager");

  // initialize the device (example: first GPU)
  Logger::log()->trace("initializing OpenCL devices...");

  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);

  if (platforms.empty()) throw std::runtime_error("No OpenCL platforms found!");

  // select the platform with the most computational resources
  // (assuming one device / per platform)
  int platform_index = 0;
  int flops_best = 0;

  Logger::log()->trace("checking device performances...");

  for (size_t kp = 0; kp < platforms.size(); kp++)
  {
    Logger::log()->trace("checking platform: {} - {}",
                         platforms[kp].getInfo<CL_PLATFORM_VENDOR>().c_str(),
                         platforms[kp].getInfo<CL_PLATFORM_NAME>().c_str());

    std::vector<cl::Device> devices;
    platforms[kp].getDevices(this->device_type, &devices);

    if (devices.empty())
    {
      Logger::log()->trace("No OpenCL devices found for this platform");
    }
    else
    {
      // estimate of the number of cores per computational unit for the
      // platform
      std::string vendor = devices[0].getInfo<CL_DEVICE_VENDOR>();
      int         cores = 1;

      if (helper_find_string_insensitive(vendor, "nvidia") ||
          helper_find_string_insensitive(vendor, "amd"))
        cores = 128;
      else if (helper_find_string_insensitive(vendor, "intel"))
        cores = 16;

      int flops = (int)devices[0].getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() *
                  (int)devices[0].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() *
                  cores;

      if (flops > flops_best)
      {
        flops_best = flops;
        platform_index = kp;
      }

      Logger::log()->trace("rating - device: {}, vendor: {}, rating: {}",
                           devices[0].getInfo<CL_DEVICE_NAME>().c_str(),
                           vendor.c_str(),
                           flops);
    }
  }

  // eventually assign the platform / device
  std::vector<cl::Device> devices;
  platforms[platform_index].getDevices(this->device_type, &devices);
  this->cl_device = devices[0];
  this->device_id = platform_index;

  Logger::log()->info("Selected OpenCL device: {}",
                      this->cl_device.getInfo<CL_DEVICE_NAME>().c_str());

  log_device_infos(this->cl_device);
}

std::map<size_t, std::string> DeviceManager::get_available_devices()
{
  std::map<size_t, std::string> device_map = {};

  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);

  if (platforms.empty()) throw std::runtime_error("No OpenCL platforms found!");

  for (size_t kp = 0; kp < platforms.size(); kp++)
  {
    std::vector<cl::Device> devices;
    platforms[kp].getDevices(this->device_type, &devices);

    if (devices.empty())
    {
      Logger::log()->trace("No OpenCL devices found for this platform");
    }
    else
    {
      std::string name = platforms[kp].getInfo<CL_PLATFORM_VENDOR>() + "/" +
                         platforms[kp].getInfo<CL_PLATFORM_NAME>() + "/" +
                         devices[0].getInfo<CL_DEVICE_NAME>();

      device_map[kp] = name;
    }
  }

  return device_map;
}

cl::Device DeviceManager::get_device() const
{
  return this->cl_device;
}

bool DeviceManager::set_device(size_t platform_id)
{
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);

  if (platforms.empty()) throw std::runtime_error("No OpenCL platforms found!");

  std::vector<cl::Device> devices;
  platforms[platform_id].getDevices(this->device_type, &devices);

  if (devices.empty())
  {
    Logger::log()->error("No OpenCL devices found for this platform");
    return false;
  }
  else
  {
    this->cl_device = devices[0];
    this->device_id = platform_id;

    Logger::log()->trace("OpenCL device: {}",
                         this->cl_device.getInfo<CL_DEVICE_NAME>().c_str());
  }

  return true;
}

void log_device_infos(cl::Device cl_device)
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

} // namespace clwrapper
