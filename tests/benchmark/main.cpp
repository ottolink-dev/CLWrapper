#include <algorithm>
#include <iomanip>
#include <iostream>
#include <vector>

#include "cl_wrapper/device_manager.hpp"
#include "cl_wrapper/kernel_manager.hpp"
#include "cl_wrapper/run.hpp"

struct BenchmarkResult
{
  std::string name;
  std::string type;
  double      score;
  float       execution_time_ms;
};

int main()
{
  std::cout << "==================================================\n";
  std::cout << "        OpenCL Device Performance Benchmark       \n";
  std::cout << "==================================================\n";

  if (!clwrapper::DeviceManager::is_ready())
  {
    std::cerr << "OpenCL DeviceManager is not ready. Exiting.\n";
    return 1;
  }

  clwrapper::DeviceManager &dm = clwrapper::DeviceManager::get_instance();
  clwrapper::KernelManager &km = clwrapper::KernelManager::get_instance();

  // Remember active device state
  size_t original_platform = dm.get_platform_id();
  size_t original_device = dm.get_device_index();

  auto cl_device_map = dm.get_all_available_devices();
  if (cl_device_map.empty())
  {
    std::cerr << "No OpenCL devices found. Exiting.\n";
    return 1;
  }

  // Math heavy math-loop kernel to measure raw compute capacity
  const std::string kernel_code =
      "__kernel void benchmark_kernel(__global const float* a, __global float* b, int iterations) {\n"
      "    int id = get_global_id(0);\n"
      "    float val = a[id];\n"
      "    for (int i = 0; i < iterations; ++i) {\n"
      "        val = val * val + 0.01f;\n"
      "        val = sin(val);\n"
      "    }\n"
      "    b[id] = val;\n"
      "}\n";

  std::vector<BenchmarkResult> results;

  const int size = 131072; // 128K elements
  const int iterations = 1000;
  std::vector<float> input(size, 0.5f);
  std::vector<float> output(size, 0.0f);

  for (auto &[coords, name] : cl_device_map)
  {
    std::cout << "Benchmarking: " << name << " ... " << std::flush;

    if (!dm.set_device(coords.first, coords.second))
    {
      std::cout << "FAILED (SetDevice)\n";
      continue;
    }

    try
    {
      // Compile the benchmark kernel for this device
      km.add_kernel(kernel_code, true, true);

      clwrapper::Run run("benchmark_kernel");
      run.bind_buffer("a", input, CL_MEM_READ_ONLY);
      run.bind_buffer("b", output, CL_MEM_WRITE_ONLY);
      run.bind_arguments(iterations);

      run.write_buffer("a");

      float elapsed = 0.0f;
      run.execute(size, &elapsed);

      run.read_buffer("b");

      // Extract device specifications
      std::string type_str = "Unknown";
      cl_device_type type = dm.get_device_type();
      if (type & CL_DEVICE_TYPE_GPU) type_str = "GPU";
      else if (type & CL_DEVICE_TYPE_CPU) type_str = "CPU";
      else if (type & CL_DEVICE_TYPE_ACCELERATOR) type_str = "Accelerator";

      double score = dm.evaluate_device(dm.get_device());

      results.push_back({name, type_str, score, elapsed});
      std::cout << "SUCCESS (" << elapsed << " ms)\n";
    }
    catch (const std::exception &e)
    {
      std::cout << "FAILED (Exception: " << e.what() << ")\n";
    }
    catch (...)
    {
      std::cout << "FAILED (Unknown exception)\n";
    }
  }

  // Restore active device state
  dm.set_device(original_platform, original_device);

  // Sort results by execution time (fastest first)
  std::sort(results.begin(), results.end(),
            [](const BenchmarkResult &r1, const BenchmarkResult &r2) {
              return r1.execution_time_ms < r2.execution_time_ms;
            });

  std::cout << "\n\n";
  std::cout << "===================================================================================\n";
  std::cout << "                                  BENCHMARK RESULTS                                \n";
  std::cout << "===================================================================================\n";
  std::cout << " " << std::left << std::setw(40) << "Device Name"
            << std::setw(15) << "Type"
            << std::setw(15) << "Eval Score"
            << std::setw(15) << "Time (ms)" << "\n";
  std::cout << "-----------------------------------------------------------------------------------\n";

  for (const auto &res : results)
  {
    std::cout << " " << std::left << std::setw(40) << res.name
              << std::setw(15) << res.type
              << std::setw(15) << std::fixed << std::setprecision(2) << res.score
              << std::setw(15) << std::fixed << std::setprecision(3) << res.execution_time_ms << "\n";
  }
  std::cout << "===================================================================================\n";

  return 0;
}
