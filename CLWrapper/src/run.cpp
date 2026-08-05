/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <chrono>

#include "cl_error_lookup.hpp"

#include "cl_wrapper/device_manager.hpp"
#include "cl_wrapper/kernel_manager.hpp"
#include "cl_wrapper/logger.hpp"
#include "cl_wrapper/run.hpp"

namespace clwrapper
{

Run::Run(const std::string &kernel_name) : kernel_name(kernel_name)
{
  Logger::log()->trace("Run::Run [{}]", this->kernel_name.c_str());

  this->queue = cl::CommandQueue(KernelManager::context(),
                                 DeviceManager::device());

  this->cl_kernel = cl::Kernel(KernelManager::program(),
                               this->kernel_name.c_str(),
                               &this->err);
  clerror::throw_opencl_error(this->err);
}

Run::~Run()
{
  this->queue.finish();
}

void Run::bind_imagef(const std::string  &id,
                      std::vector<float> &vector,
                      int                 width,
                      int                 height,
                      Direction           direction)
{
  Image2D img;

  img.vector_ref = static_cast<void *>(vector.data());
  img.width = width;
  img.height = height;

  if (direction == Direction::IN)
    img.cl_image = cl::Image2D(KernelManager::context(),
                               CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               cl::ImageFormat(CL_R, CL_FLOAT),
                               width,
                               height,
                               0,
                               (void *)img.vector_ref,
                               &this->err);
  else
    img.cl_image = cl::Image2D(KernelManager::context(),
                               CL_MEM_WRITE_ONLY,
                               cl::ImageFormat(CL_R, CL_FLOAT),
                               width,
                               height,
                               0,
                               nullptr,
                               &this->err);

  clerror::throw_opencl_error(this->err);

  this->err = this->cl_kernel.setArg(this->arg_count++, img.cl_image);
  clerror::throw_opencl_error(this->err);

  this->images_2d[id] = img;
}

void Run::bind_imagef(const std::string  &id,
                      std::vector<float> &vector,
                      int                 width,
                      int                 height,
                      bool                is_out)
{
  Direction direction = is_out ? Direction::OUT : Direction::IN;
  this->bind_imagef(id, vector, width, height, direction);
}

void Run::bind_imagef(const std::string        &id,
                      const std::vector<float> &vector,
                      int                       width,
                      int                       height,
                      bool                      is_out)
{
  this->bind_imagef(id,
                    const_cast<std::vector<float> &>(vector),
                    width,
                    height,
                    is_out);
}

void Run::execute(int total_elements, float *p_elapsed_time)
{
  this->queue.flush();

  int bsize = 8;
  int gsize = ((total_elements + bsize - 1) / bsize) * bsize;

  const cl::NDRange global_work_size(gsize);

  this->err = this->queue.enqueueNDRangeKernel(this->cl_kernel,
                                               cl::NullRange,
                                               global_work_size,
                                               cl::NullRange);
  clerror::throw_opencl_error(this->err);

  auto t0 = std::chrono::high_resolution_clock::now();

  this->err = this->queue.finish();
  clerror::throw_opencl_error(this->err);

  if (p_elapsed_time)
  {
    auto t1 = std::chrono::high_resolution_clock::now();

    *p_elapsed_time =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() *
        1e-6f;
  }
}

void Run::execute(const std::vector<int> &global_range_2d,
                  float                  *p_elapsed_time)
{
  this->queue.flush();

  int bsize = 8;
  int gsize_x = ((global_range_2d[0] + bsize - 1) / bsize) * bsize;
  int gsize_y = ((global_range_2d[1] + bsize - 1) / bsize) * bsize;

  const cl::NDRange global_work_size(gsize_x, gsize_y);

  this->err = this->queue.enqueueNDRangeKernel(this->cl_kernel,
                                               cl::NullRange,
                                               global_work_size,
                                               cl::NullRange);
  clerror::throw_opencl_error(this->err);

  auto t0 = std::chrono::high_resolution_clock::now();

  this->err = this->queue.flush();
  clerror::throw_opencl_error(this->err);

  if (p_elapsed_time)
  {
    auto t1 = std::chrono::high_resolution_clock::now();

    *p_elapsed_time =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() *
        1e-6f;
  }
}

void Run::read_buffer(const std::string &id)
{
  if (this->buffers.find(id) != this->buffers.end())
  {
    this->err = this->queue.enqueueReadBuffer(this->buffers[id].cl_buffer,
                                              CL_TRUE,
                                              0,
                                              this->buffers[id].size,
                                              this->buffers[id].vector_ref);
    clerror::throw_opencl_error(this->err);
  }
  else
  {
    Logger::log()->error("unknown buffer id: [{}]", id.c_str());
  }
}

void Run::read_imagef(const std::string &id)
{
  if (this->images_2d.find(id) != this->images_2d.end())
  {
    cl::array<size_t, 3> origin = {0, 0, 0};
    cl::array<size_t, 3> region = {(size_t)this->images_2d[id].width,
                                   (size_t)this->images_2d[id].height,
                                   1};

    this->err = this->queue.enqueueReadImage(this->images_2d[id].cl_image,
                                             CL_TRUE,
                                             origin,
                                             region,
                                             0,
                                             0,
                                             this->images_2d[id].vector_ref);
    clerror::throw_opencl_error(this->err);
  }
  else
  {
    Logger::log()->error("unknown 2D imagef id: [{}]", id.c_str());
  }
}

void Run::reset_argcount()
{
  this->arg_count = 0;
}

void Run::write_buffer(const std::string &id)
{
  if (this->buffers.find(id) != this->buffers.end())
  {
    this->err = this->queue.enqueueWriteBuffer(this->buffers[id].cl_buffer,
                                               CL_TRUE,
                                               0,
                                               this->buffers[id].size,
                                               this->buffers[id].vector_ref);
    clerror::throw_opencl_error(this->err);
  }
  else
  {
    Logger::log()->error("unknown buffer id: [{}]", id.c_str());
  }
}

void Run::write_imagef(const std::string &id)
{
  if (this->images_2d.find(id) != this->images_2d.end())
  {
    cl::array<size_t, 3> origin = {0, 0, 0};
    cl::array<size_t, 3> region = {(size_t)this->images_2d[id].width,
                                   (size_t)this->images_2d[id].height,
                                   1};

    this->err = this->queue.enqueueWriteImage(this->images_2d[id].cl_image,
                                              CL_TRUE,
                                              origin,
                                              region,
                                              0,
                                              0,
                                              this->images_2d[id].vector_ref);
    clerror::throw_opencl_error(this->err);
  }
  else
  {
    Logger::log()->error("unknown 2D imagef id: [{}]", id.c_str());
  }
}

} // namespace clwrapper
