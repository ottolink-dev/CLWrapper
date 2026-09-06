# CLWrapper

This library provides a C++ wrapper for OpenCL, simplifying tasks like context creation, device management, program compilation, and kernel execution. It abstracts common OpenCL tasks and allows to integrate GPU-based parallel computing into C++ applications. Image buffer bindings are also provided.

## Getting the Sources

Clone the repository using `git`:

```bash
git clone git@github.com:otto-link/CLWrapper.git
cd CLWrapper
git submodule update --init --recursive
```

## Build and Test

### Prerequisites
- A C++ compiler with C++17 support or higher.
- CMake version 3.15 or newer.

### Build Instructions
1. Create a build directory:
   ```bash
   mkdir build && cd build
   ```
2. Run CMake to configure the build:
   ```bash
   cmake ..
   ```
3. Build the project using `make`:
   ```bash
   make
   ```

### Run Tests
After building, you can run the tests to verify the build:
```bash
bin/test_clwrapper
```

## CMake Integration

To integrate CLWrapper into your CMake-based project, follow these steps:

1. Add the library to your project's `CMakeLists.txt`:
   ```cmake
   add_subdirectory(CLWrapper)
   target_link_libraries(your_project_target clwrapper)
   ```
2. Link the `clwrapper` library to your target, as shown above.

## Usage Example

### Basic Example
```cpp
#include <iostream>

#include "cl_wrapper.hpp"

int main()
{
  const std::string code =
#include "add.cl"
      ;

  clwrapper::KernelManager::get_instance().add_kernel(code);

  auto run = clwrapper::Run("add_kernel_with_args");

  int                n = 11;
  std::vector<float> a(n, 1.f);
  std::vector<float> b(n, 2.f);
  std::vector<float> c(n); // output

  run.bind_buffer<float>("a", a);
  run.bind_buffer<float>("b", b);
  run.bind_buffer<float>("c", c);

  float p1 = 1.f;
  float p2 = 2.f;
  int   p3 = 1;
  run.bind_arguments(n, p1, p2, p3);

  run.write_buffer("a");
  run.write_buffer("b");

  run.execute(n);

  run.read_buffer("c");

  for (size_t k = 0; k < c.size(); ++k)
    std::cout << k << ": " << c[k] << "\n";

  return 0;
}
```

The file `add.cl` contains:
```C++
R""(
kernel void add_kernel(global float *A,
                       global float *B,
                       global float *C,
                       const int     n)
{
  const uint i = get_global_id(0);

  if (i >= n) return;

  C[i] = A[i] + B[i];
}

kernel void add_kernel_with_args(global float *A,
                                 global float *B,
                                 global float *C,
                                 const int     n,
                                 const float   p1,
                                 const float   p2,
                                 const int     p3)
{
  const uint i = get_global_id(0);

  if (i >= n) return;

  C[i] = A[i] + B[i] + p1 + p2 + p3;
}
)""
```

### Keeping data on the device across kernels

Several `Run` objects can share one in-order command queue and the same
device images. Combined with `execute_async`, an iterative multi-kernel loop
never touches host memory until the end:

```cpp
clwrapper::Run pass_a("kernel_a");
pass_a.bind_imagef("x", x, nx, ny, clwrapper::Direction::INOUT);
pass_a.bind_imagef("y", y, nx, ny, clwrapper::Direction::INOUT);

clwrapper::Run pass_b("kernel_b", pass_a.get_queue());
pass_b.bind_image2d("y", pass_a.get_image2d("y")); // same device image
pass_b.bind_image2d("x", pass_a.get_image2d("x"));

for (int it = 0; it < iterations; ++it)
{
  pass_a.execute_async({nx, ny}); // x -> y
  pass_b.execute_async({nx, ny}); // y -> x
}
pass_b.finish();
pass_b.read_imagef("x");
```

`Direction::INOUT` creates a read-write image initialised from the host.
Use `set_argument(pos, run.get_image2d("id").cl_image)` to swap ping-pong
buffers between iterations without re-uploading.

## Contributing

If you find any incorrect or missing error codes, please use the [GitHub Issues](https://github.com/otto-link/CLErrorLookup/issues) to propose modifications. Contributions are always welcome and help ensure the accuracy and usefulness of the library.

## License

This project is licensed under the GPL-3.0 License. See the `LICENSE` file for details.
