/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */

/**
 * @file device_manager.hpp
 * @author Otto Link (otto.link.bv@gmail.com)
 * @brief Singleton manager to handle OpenCL device querying, selection, and
 * metadata retrieval.
 * @copyright Copyright (c) 2025
 */
#pragma once
#include <map>
#include <shared_mutex>
#include <string>
#include <utility>

#include <CL/opencl.hpp>

namespace clwrapper
{

/**
 * @class DeviceManager
 * @brief Handles the initialization, management, and selection of OpenCL
 * platforms and devices.
 *
 * Provides features for automatic optimal device discovery, manual selection of
 * platforms/devices, and querying detailed hardware and execution limits.
 */
class DeviceManager
{
public:
  /**
   * @name Singleton Access
   * @{
   */

  /**
   * @brief Gets the global singleton instance of the DeviceManager.
   * @return A reference to the active DeviceManager instance.
   */
  static DeviceManager &get_instance();

  /**
   * @brief Access the OpenCL device attached to the active DeviceManager
   * instance.
   * @return The currently active cl::Device.
   */
  static cl::Device device();

  /**
   * @brief Checks if the DeviceManager singleton is ready to use and
   * initialized.
   * @return True if initialized successfully; false otherwise.
   */
  static bool is_ready();

  /** @} */

  /**
   * @name Device Information and Querying
   * @{
   */

  /**
   * @brief Lists all the available devices (first device of each platform).
   * @return A map mapping platform IDs to their respective primary device
   * names.
   */
  std::map<size_t, std::string> get_available_devices();

  /**
   * @brief Lists all devices on all platforms with explicit coordinates.
   * @return A map mapping pairs of {platform_id, device_index} to device names.
   */
  std::map<std::pair<size_t, size_t>, std::string> get_all_available_devices();

  /**
   * @brief Access the OpenCL device.
   * @return The active cl::Device.
   */
  cl::Device get_device() const;

  /**
   * @brief Get the ID of the active platform.
   * @return The active platform ID.
   */
  size_t get_platform_id() const;

  /**
   * @brief Get the index of the active device on the platform.
   * @return The active device index.
   */
  size_t get_device_index() const;

  /**
   * @brief Get the ID of the active platform (backward compatibility alias for
   * get_platform_id).
   * @return The active platform ID.
   */
  size_t get_device_id() const;

  /** @} */

  /**
   * @name Device Selection and Configuration
   * @{
   */

  /**
   * @brief Set the active platform (defaults to device 0 on that platform).
   * @param platform_id The platform ID to activate.
   * @return True if successfully set; false otherwise.
   */
  bool set_device(size_t platform_id);

  /**
   * @brief Set the active device with explicit platform and device index
   * coordinates.
   * @param platform_id The platform ID to use.
   * @param device_index The device index on that platform.
   * @return True if successfully set; false otherwise.
   */
  bool set_device(size_t platform_id, size_t device_index);

  /**
   * @brief Configures the type of device filtered during platform checks (CPU,
   * GPU, etc.).
   * @param new_device_type The OpenCL device type filter.
   */
  void set_device_type(cl_device_type new_device_type);

  /** @} */

  /**
   * @name Active Device Metadata Accessors
   * @{
   */

  /**
   * @brief Gets the name of the active device.
   */
  std::string get_device_name() const;

  /**
   * @brief Gets the vendor of the active device.
   */
  std::string get_device_vendor() const;

  /**
   * @brief Gets the OpenCL version supported by the active device.
   */
  std::string get_device_version() const;

  /**
   * @brief Gets the type of the active device.
   */
  cl_device_type get_device_type() const;

  /**
   * @brief Gets the number of parallel compute units on the active device.
   */
  cl_uint get_max_compute_units() const;

  /**
   * @brief Gets the maximum number of work-items in a work-group executing a
   * kernel.
   */
  size_t get_max_work_group_size() const;

  /**
   * @brief Gets the size of global device memory in bytes.
   */
  cl_ulong get_global_mem_size() const;

  /**
   * @brief Gets the size of local device memory in bytes.
   */
  cl_ulong get_local_mem_size() const;

  /**
   * @brief Computes a robustness capability score for a given OpenCL device.
   *
   * Scores are calculated based on device type (GPU > Accelerator > CPU),
   * discrete vs unified memory, compute capacity (units * clock speed), and
   * total global memory size.
   *
   * @param device The cl::Device to evaluate.
   * @return A numerical score representing the capability level of the device.
   */
  double evaluate_device(const cl::Device &device) const;

  /** @} */

  /**
   * @name Active Platform Metadata Accessors
   * @{
   */

  /**
   * @brief Gets the name of the active platform.
   */
  std::string get_platform_name() const;

  /**
   * @brief Gets the vendor of the active platform.
   */
  std::string get_platform_vendor() const;

  /**
   * @brief Gets the OpenCL version supported by the active platform.
   */
  std::string get_platform_version() const;

  /** @} */

private:
  cl::Device cl_device;

  size_t platform_id = 0;
  size_t device_index = 0;

  // allowed device type (CL_DEVICE_TYPE_ALL | GPU | CPU)
  cl_device_type device_type = CL_DEVICE_TYPE_ALL;

  // Mutex protecting access to device selection and state
  mutable std::shared_mutex state_mutex;

  // Private constructor
  DeviceManager();

  // Internal helper to search and select the optimal device (assumes
  // state_mutex is already locked for writing)
  void select_optimal_device();

  // Delete copy/move constructors and assignment operators to enforce singleton
  DeviceManager(const DeviceManager &) = delete;
  DeviceManager &operator=(const DeviceManager &) = delete;
  DeviceManager(DeviceManager &&) = delete;
  DeviceManager &operator=(DeviceManager &&) = delete;
};

/**
 * @brief Logs detailed diagnostic information about an OpenCL device.
 * @param cl_device The cl::Device to query and log.
 */
void log_device_infos(cl::Device cl_device);

} // namespace clwrapper
