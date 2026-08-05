/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <mutex>
#include <shared_mutex>

#include "cl_error_lookup.hpp"

#include "cl_wrapper/device_manager.hpp"
#include "cl_wrapper/kernel_manager.hpp"
#include "cl_wrapper/logger.hpp"

namespace clwrapper
{

KernelManager::KernelManager()
{
  this->build_program();
}

void KernelManager::add_kernel(const std::string &kernel_sources,
                               bool               clear_sources)
{
  std::unique_lock<std::shared_mutex> lock(this->state_mutex);
  if (clear_sources)
    this->full_sources = kernel_sources;
  else
    this->full_sources += kernel_sources;

  this->build_program();
}

void KernelManager::build_program()
{
  Logger::log()->trace("loading kernel sources");

  if (this->full_sources.length() > 0)
  {
    try
    {
      cl::Device cl_device = clwrapper::DeviceManager::device();
      this->cl_context = cl::Context({cl_device});

      cl::Program::Sources sources;
      sources.push_back(
          {this->full_sources.c_str(), this->full_sources.length()});

      Logger::log()->trace("building OpenCL kernels");
      Logger::log()->trace("build options: {}", this->build_options);

      this->cl_program = cl::Program(this->cl_context, sources);
      int err = this->cl_program.build({cl_device}, this->build_options.c_str());

      if (err != 0)
      {
        std::string build_log = this->cl_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(cl_device);
        Logger::log()->critical("build error: {}\nOpenCL compiler says:\n----------------------------------------------\n{}\n----------------------------------------------", err, build_log);
        clerror::throw_opencl_error(err);
      }

      std::string kernel_names = this->cl_program.getInfo<CL_PROGRAM_KERNEL_NAMES>();
      Logger::log()->trace("available kernels: {}", kernel_names.c_str());
    }
    catch (const std::exception &e)
    {
      Logger::log()->error("Exception during build_program: {}", e.what());
      throw;
    }
    catch (...)
    {
      Logger::log()->error("Unknown exception during build_program");
      throw;
    }
  }
  else
  {
    Logger::log()->trace("program building skipped, kernel sources are empty");
  }
}

void KernelManager::clear_sources()
{
  std::unique_lock<std::shared_mutex> lock(this->state_mutex);
  this->full_sources = "";
}

cl::Context KernelManager::context()
{
  return KernelManager::get_instance().get_context();
}

std::string KernelManager::get_build_options() const
{
  std::shared_lock<std::shared_mutex> lock(this->state_mutex);
  return this->build_options;
}

cl::Context KernelManager::get_context() const
{
  std::shared_lock<std::shared_mutex> lock(this->state_mutex);
  return this->cl_context;
}

std::string KernelManager::get_full_sources() const
{
  std::shared_lock<std::shared_mutex> lock(this->state_mutex);
  return this->full_sources;
}

KernelManager &KernelManager::get_instance()
{
  static KernelManager instance;
  return instance;
}

cl::Program KernelManager::get_program() const
{
  std::shared_lock<std::shared_mutex> lock(this->state_mutex);
  return this->cl_program;
}

cl::Program KernelManager::program()
{
  return KernelManager::get_instance().get_program();
}

void KernelManager::set_build_options(const std::string &new_build_options)
{
  std::unique_lock<std::shared_mutex> lock(this->state_mutex);
  this->build_options = new_build_options;
}

} // namespace clwrapper
