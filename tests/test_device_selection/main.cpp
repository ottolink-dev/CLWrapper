/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <iostream>

#include "cl_wrapper.hpp"

int main()
{
  auto &dm = clwrapper::DeviceManager::get_instance();
  size_t original_platform = dm.get_platform_id();
  size_t original_device = dm.get_device_index();

  std::map<std::pair<size_t, size_t>, std::string> cl_device_map =
      dm.get_all_available_devices();

  std::cout << "================================================================================\n";
  std::cout << "AVAILABLE OPENCL DEVICES & SPECIFICATIONS\n";
  std::cout << "================================================================================\n";

  for (auto &[coords, name] : cl_device_map)
  {
    if (dm.set_device(coords.first, coords.second))
    {
      std::string type_str = "Unknown";
      cl_device_type type = dm.get_device_type();
      if (type & CL_DEVICE_TYPE_GPU) type_str = "GPU";
      else if (type & CL_DEVICE_TYPE_CPU) type_str = "CPU";
      else if (type & CL_DEVICE_TYPE_ACCELERATOR) type_str = "Accelerator";

      double score = dm.evaluate_device(dm.get_device());
      double global_mem_gb = static_cast<double>(dm.get_global_mem_size()) / (1024 * 1024 * 1024);
      double local_mem_kb = static_cast<double>(dm.get_local_mem_size()) / 1024;

      std::cout << "Platform: " << coords.first << " | Device: " << coords.second << "\n";
      std::cout << "  Name:         " << dm.get_device_name() << "\n";
      std::cout << "  Vendor:       " << dm.get_device_vendor() << "\n";
      std::cout << "  Version:      " << dm.get_device_version() << "\n";
      std::cout << "  Type:         " << type_str << "\n";
      std::cout << "  Compute CUs:  " << dm.get_max_compute_units() << "\n";
      std::cout << "  Max WG Size:  " << dm.get_max_work_group_size() << "\n";
      std::cout << "  Global Mem:   " << global_mem_gb << " GB\n";
      std::cout << "  Local Mem:    " << local_mem_kb << " KB\n";
      std::cout << "  Score:        " << score << "\n";
      std::cout << "--------------------------------------------------------------------------------\n";
    }
  }

  // Restore the original active device
  dm.set_device(original_platform, original_device);

  std::cout << "\nSelected Default Device: " << dm.get_device_name() << "\n";
  std::cout << "================================================================================\n";

  // --- execute the same kernel on each device

  const std::string code =
#include "add.cl"
      ;

  // add the source (will be build for the current or default device)
  clwrapper::KernelManager::get_instance().add_kernel(code);

  for (auto &[coords, name] : cl_device_map)
  {
    std::cout << "\n\n--- Running kernel on " << name << " ---\n\n";

    if (clwrapper::DeviceManager::get_instance().set_device(coords.first, coords.second))
    {
      // program needs to be rebuild for the current device
      clwrapper::KernelManager::get_instance().build_program();

      // reminder is "standard" run execution
      auto run = clwrapper::Run("add_kernel");

      int                n = 9;
      std::vector<float> a(n, 1.f);
      std::vector<float> b(n, 2.f);
      std::vector<float> c(n); // output

      run.bind_buffer<float>("a", a);
      run.bind_buffer<float>("b", b);
      run.bind_buffer<float>("c", c);
      run.write_buffer("a");
      run.write_buffer("b");

      run.execute(n);
      run.read_buffer("c");

      for (auto &v : c)
        std::cout << v << "\n";
    }
  }

  return 0;
}
