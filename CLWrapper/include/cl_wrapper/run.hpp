/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */

/**
 * @file run.hpp
 * @author Otto Link (otto.link.bv@gmail.com)
 * @brief
 *
 * @copyright Copyright (c) 2025
 */
#pragma once
#include <map>

#include <CL/opencl.hpp>

#include "cl_error_lookup.hpp"

#include "cl_wrapper/kernel_manager.hpp"

namespace clwrapper
{

// helper
template <typename T> size_t vector_sizeof(const typename std::vector<T> &v)
{
  return sizeof(T) * v.size();
}

struct Buffer
{
  cl::Buffer cl_buffer;
  void      *vector_ref;
  size_t     size;
};

struct Image2D
{
  cl::Image2D cl_image;
  void       *vector_ref;
  int         width;
  int         height;
};

enum Direction
{
  IN,
  OUT
};

// class
class Run
{
public:
  Run(const std::string &kernel_name);

  ~Run();

  template <typename T> void bind_arguments(T arg)
  {
    err = this->cl_kernel.setArg(this->arg_count++, arg);
    clerror::throw_opencl_error(err);
  }

  template <typename T> void set_argument(int arg_pos, T arg)
  {
    err = this->cl_kernel.setArg(arg_pos, arg);
    clerror::throw_opencl_error(err);
  }

  template <typename... Args> void bind_arguments(Args... args)
  {
    // Expand the parameter pack and call fct for each argument
    (this->bind_arguments(args), ...);
  }

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
                                  &err);
    clerror::throw_opencl_error(err);

    err = this->cl_kernel.setArg(this->arg_count++, buffer.cl_buffer);
    clerror::throw_opencl_error(err);

    this->buffers[id] = buffer;
  }

  template <typename T>
  void bind_buffer(const std::string    &id,
                   const std::vector<T> &vector,
                   cl_mem_flags          flags = CL_MEM_READ_WRITE)
  {
    this->bind_buffer<T>(id, const_cast<std::vector<T> &>(vector), flags);
  }

  // data are copied at binding
  void bind_imagef(const std::string  &id,
                   std::vector<float> &vector,
                   int                 width,
                   int                 height,
                   Direction           direction);

  void bind_imagef(const std::string  &id,
                   std::vector<float> &vector,
                   int                 width,
                   int                 height,
                   bool                is_out = false);

  void bind_imagef(const std::string        &id,
                   const std::vector<float> &vector,
                   int                       width,
                   int                       height,
                   bool                      is_out = false);

  void execute(int total_elements, float *p_elapsed_time = nullptr);

  void execute(const std::vector<int> &global_range_2d,
               float                  *p_elapsed_time = nullptr);

  void read_buffer(const std::string &id);

  void read_imagef(const std::string &id);

  void reset_argcount()
  {
    this->arg_count = 0;
  }

  void write_buffer(const std::string &id);

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
