/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */

/**
 * @file kernel_manager.hpp
 * @author Otto Link (otto.link.bv@gmail.com)
 * @brief Singleton manager to handle OpenCL context creation, program compilation, and kernel building.
 * @copyright Copyright (c) 2025
 */
#pragma once
#include <shared_mutex>
#include <string>

#include <CL/opencl.hpp>

namespace clwrapper
{

/**
 * @class KernelManager
 * @brief Manages OpenCL context, compilation, and registration of kernel sources.
 * 
 * Handles the compilation of OpenCL program objects, tracking available kernels,
 * and maintaining program build options in a thread-safe manner.
 */
class KernelManager
{
public:
  /**
   * @name Singleton Access
   * @{
   */

  /**
   * @brief Gets the global singleton instance of the KernelManager.
   * @return A reference to the active KernelManager instance.
   */
  static KernelManager &get_instance();

  /**
   * @brief Gets the OpenCL context attached to the active KernelManager instance.
   * @return The active cl::Context.
   */
  static cl::Context context();

  /**
   * @brief Gets the OpenCL program attached to the active KernelManager instance.
   * @return The active cl::Program.
   */
  static cl::Program program();

  /** @} */

  /**
   * @name Program Compilation and Source Configuration
   * @{
   */

  /**
   * @brief Appends or replaces OpenCL kernel source code and triggers a rebuild.
   * @param kernel_sources The OpenCL source code string to register.
   * @param clear_sources If true, replaces existing sources; if false, appends them.
   */
  void add_kernel(const std::string &kernel_sources,
                  bool               clear_sources = false);

  /**
   * @brief Compiles the registered OpenCL program sources with configured build options.
   */
  void build_program();

  /**
   * @brief Clears all registered OpenCL source code.
   */
  void clear_sources();

  /**
   * @brief Configures compiler build options passed to the OpenCL compiler.
   * @param new_build_options Compiler options string (e.g. "-cl-mad-enable").
   */
  void set_build_options(const std::string &new_build_options);

  /** @} */

  /**
   * @name Instance Member Accessors
   * @{
   */

  /**
   * @brief Access the OpenCL context.
   * @return The active cl::Context.
   */
  cl::Context get_context() const;

  /**
   * @brief Access the compiled OpenCL program.
   * @return The active cl::Program.
   */
  cl::Program get_program() const;

  /**
   * @brief Access the full registered OpenCL source code.
   * @return The source code string.
   */
  std::string get_full_sources() const;

  /**
   * @brief Access the current OpenCL compiler build options.
   * @return The build options string.
   */
  std::string get_build_options() const;

  /** @} */

private:
  cl::Program cl_program;

  cl::Context cl_context;

  std::string full_sources = "";

  std::string build_options = "";

  // Mutex protecting access to compilation and program state
  mutable std::shared_mutex state_mutex;

  // Private constructor
  KernelManager();

  // Delete copy/move constructors and assignment operators to enforce singleton
  KernelManager(const KernelManager &) = delete;
  KernelManager &operator=(const KernelManager &) = delete;
  KernelManager(KernelManager &&) = delete;
  KernelManager &operator=(KernelManager &&) = delete;
};

} // namespace clwrapper
