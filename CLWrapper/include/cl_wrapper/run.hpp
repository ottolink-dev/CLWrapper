/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */

/**
 * @file run.hpp
 * @author Otto Link (otto.link.bv@gmail.com)
 * @brief Handles OpenCL kernel execution, argument bindings, and memory
 * transfers.
 * @copyright Copyright (c) 2025
 */
#pragma once
#include <map>
#include <string>
#include <vector>

#include <CL/opencl.hpp>

#include "cl_error_lookup.hpp"
#include "cl_wrapper/kernel_manager.hpp"

namespace clwrapper
{

/**
 * @brief Helper utility to calculate the byte size of a standard vector.
 * @tparam T The element type of the vector.
 * @param v The vector reference.
 * @return Total byte size of the vector elements.
 */
template <typename T> size_t vector_sizeof(const typename std::vector<T> &v)
{
  return sizeof(T) * v.size();
}

/**
 * @struct Buffer
 * @brief Wrap container for an OpenCL 1D memory buffer and its host reference.
 */
struct Buffer
{
  cl::Buffer cl_buffer;  ///< The OpenCL buffer object.
  void      *vector_ref; ///< Host pointer reference.
  size_t     size;       ///< Total buffer size in bytes.
};

/**
 * @struct Image2D
 * @brief Wrap container for an OpenCL 2D image and its host reference.
 */
struct Image2D
{
  cl::Image2D cl_image;   ///< The OpenCL 2D image object.
  void       *vector_ref; ///< Host pointer reference.
  int         width;      ///< Width of the 2D image.
  int         height;     ///< Height of the 2D image.
};

/**
 * @enum Direction
 * @brief Declares the data flow direction relative to the host for image
 * bindings.
 */
enum Direction
{
  IN, ///< Input data flow (read-only on device).
  OUT ///< Output data flow (write-only on device).
};

/**
 * @class Run
 * @brief Handles single execution instances of a specific OpenCL kernel.
 *
 * Manages binding simple arguments, 1D buffers, and 2D images, triggering
 * NDRange kernel execution, and transferring data between device and host
 * memory.
 */
class Run
{
public:
  /**
   * @brief Constructs a Run instance for a specific kernel.
   * @param kernel_name The name of the kernel to bind and run.
   */
  Run(const std::string &kernel_name);

  /**
   * @brief Destructor. Automatically flushes and finishes active command
   * queues.
   */
  ~Run();

  /**
   * @brief Binds a single value argument to the next available kernel argument
   * index.
   * @tparam T The argument type.
   * @param arg The value to bind.
   */
  template <typename T> void bind_arguments(T arg)
  {
    this->err = this->cl_kernel.setArg(this->arg_count++, arg);
    clerror::throw_opencl_error(this->err);
  }

  /**
   * @brief Binds multiple value arguments sequentially.
   * @tparam Args Variadic parameter pack.
   * @param args Arguments to bind.
   */
  template <typename... Args> void bind_arguments(Args... args)
  {
    (this->bind_arguments(args), ...);
  }

  /**
   * @brief Explicitly overrides a specific argument index with a value.
   * @tparam T The argument type.
   * @param arg_pos Zero-indexed argument position.
   * @param arg The value to set.
   */
  template <typename T> void set_argument(int arg_pos, T arg)
  {
    this->err = this->cl_kernel.setArg(arg_pos, arg);
    clerror::throw_opencl_error(this->err);
  }

  /**
   * @brief Binds a mutable host vector as a 1D OpenCL buffer.
   * @tparam T The element type.
   * @param id A unique string ID to identify this buffer.
   * @param vector Ref to the host vector.
   * @param flags OpenCL memory flags (e.g. CL_MEM_READ_WRITE).
   */
  template <typename T>
  void bind_buffer(const std::string &id,
                   std::vector<T>    &vector,
                   cl_mem_flags       flags = CL_MEM_READ_WRITE)
  {
    Buffer buffer;

    buffer.vector_ref = static_cast<void *>(vector.data());
    buffer.size = vector_sizeof<T>(vector);
    buffer.cl_buffer = cl::Buffer(KernelManager::context(),
                                  flags,
                                  buffer.size,
                                  nullptr,
                                  &this->err);
    clerror::throw_opencl_error(this->err);

    this->err = this->cl_kernel.setArg(this->arg_count++, buffer.cl_buffer);
    clerror::throw_opencl_error(this->err);

    this->buffers[id] = buffer;
  }

  /**
   * @brief Binds a const host vector as a 1D OpenCL buffer (backward
   * compatibility overload).
   * @tparam T The element type.
   * @param id A unique string ID.
   * @param vector Ref to the const host vector.
   * @param flags OpenCL memory flags.
   */
  template <typename T>
  void bind_buffer(const std::string    &id,
                   const std::vector<T> &vector,
                   cl_mem_flags          flags = CL_MEM_READ_WRITE)
  {
    this->bind_buffer<T>(id, const_cast<std::vector<T> &>(vector), flags);
  }

  /**
   * @brief Binds a host vector as a 2D float image with a specified direction.
   */
  void bind_imagef(const std::string  &id,
                   std::vector<float> &vector,
                   int                 width,
                   int                 height,
                   Direction           direction);

  /**
   * @brief Binds a host vector as a 2D float image (backward compatibility
   * overload).
   */
  void bind_imagef(const std::string  &id,
                   std::vector<float> &vector,
                   int                 width,
                   int                 height,
                   bool                is_out = false);

  /**
   * @brief Binds a const host vector as a 2D float image (backward
   * compatibility overload).
   */
  void bind_imagef(const std::string        &id,
                   const std::vector<float> &vector,
                   int                       width,
                   int                       height,
                   bool                      is_out = false);

  /**
   * @brief Executes the kernel over a 1D range.
   * @param total_elements The total size of work-items.
   * @param p_elapsed_time Optional out parameter to receive execution duration
   * in milliseconds.
   */
  void execute(int total_elements, float *p_elapsed_time = nullptr);

  /**
   * @brief Executes the kernel over a 2D range.
   * @param global_range_2d The 2D dimensions of work-items (width, height).
   * @param p_elapsed_time Optional out parameter to receive execution duration
   * in milliseconds.
   */
  void execute(const std::vector<int> &global_range_2d,
               float                  *p_elapsed_time = nullptr);

  /**
   * @brief Reads data back from the specified device buffer to its registered
   * host vector.
   * @param id The unique string ID of the buffer.
   */
  void read_buffer(const std::string &id);

  /**
   * @brief Reads data back from the specified 2D device image to its registered
   * host vector.
   * @param id The unique string ID of the image.
   */
  void read_imagef(const std::string &id);

  /**
   * @brief Resets the internal bound arguments counter back to zero.
   */
  void reset_argcount();

  /**
   * @brief Writes data from the registered host vector to the specified device
   * buffer.
   * @param id The unique string ID of the buffer.
   */
  void write_buffer(const std::string &id);

  /**
   * @brief Writes data from the registered host vector to the specified 2D
   * device image.
   * @param id The unique string ID of the image.
   */
  void write_imagef(const std::string &id);

private:
  std::string kernel_name;

  cl::CommandQueue queue;

  cl::Kernel cl_kernel;

  int arg_count = 0;

  std::map<std::string, Buffer> buffers;

  std::map<std::string, Image2D> images_2d;

  int err = 0;
};

} // namespace clwrapper
