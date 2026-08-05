#include "cl_wrapper/device_manager.hpp"
#include "cl_wrapper/kernel_manager.hpp"
#include "cl_wrapper/run.hpp"
#include <gtest/gtest.h>

using namespace clwrapper;

TEST(RunTest, BufferBindingAndExecution)
{
  KernelManager &km = KernelManager::get_instance();
  km.clear_sources();

  const std::string code = "__kernel void vector_add(__global const float* a, "
                           "__global const float* b, __global float* c) {\n"
                           "    int id = get_global_id(0);\n"
                           "    c[id] = a[id] + b[id];\n"
                           "}\n";

  km.add_kernel(code, true, true);

  clwrapper::Run     run("vector_add");
  int                n = 10;
  std::vector<float> a(n, 1.5f);
  std::vector<float> b(n, 2.5f);
  std::vector<float> c(n, 0.0f);

  run.bind_buffer("a", a);
  run.bind_buffer("b", b);
  run.bind_buffer("c", c);

  run.write_buffer("a");
  run.write_buffer("b");

  float elapsed = 0.0f;
  EXPECT_NO_THROW(run.execute(n, &elapsed));
  EXPECT_GT(elapsed, 0.0f);

  run.read_buffer("c");

  for (int i = 0; i < n; ++i)
  {
    EXPECT_FLOAT_EQ(c[i], 4.0f);
  }
}

TEST(RunTest, ImageBindingAndExecution)
{
  KernelManager &km = KernelManager::get_instance();
  km.clear_sources();

  const std::string img_kernel_code =
      "__kernel void img_test(__read_only image2d_t src, __write_only "
      "image2d_t dest) {\n"
      "    int x = get_global_id(0);\n"
      "    int y = get_global_id(1);\n"
      "    if (x < get_image_width(src) && y < get_image_height(src)) {\n"
      "        const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | "
      "CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n"
      "        float4 val = read_imagef(src, sampler, (int2)(x, y));\n"
      "        write_imagef(dest, (int2)(x, y), val * 2.0f);\n"
      "    }\n"
      "}\n";

  km.add_kernel(img_kernel_code, true, true);

  clwrapper::Run run("img_test");
  int            width = 4;
  int            height = 3;

  std::vector<float> src_data(width * height, 3.0f);
  std::vector<float> dest_data(width * height, 0.0f);

  run.bind_imagef("src", src_data, width, height, Direction::IN);
  run.bind_imagef("dest", dest_data, width, height, Direction::OUT);

  // Note: the kernel doesn't take width and height as arguments, so we do not
  // bind them. Reset arg count to be sure.
  run.reset_argcount();
  run.bind_imagef("src", src_data, width, height, Direction::IN);
  run.bind_imagef("dest", dest_data, width, height, Direction::OUT);

  EXPECT_NO_THROW(run.execute({width, height}));

  run.read_imagef("dest");

  for (int i = 0; i < width * height; ++i)
  {
    EXPECT_FLOAT_EQ(dest_data[i], 6.0f);
  }
}

TEST(RunTest, ArgumentValidationTest)
{
  KernelManager &km = KernelManager::get_instance();
  km.clear_sources();

  const std::string code = "__kernel void simple_scalar_add(__global const "
                           "float* a, float factor, __global float* b) {\n"
                           "    int id = get_global_id(0);\n"
                           "    b[id] = a[id] + factor;\n"
                           "}\n";

  km.add_kernel(code, true, true);

  clwrapper::Run     run("simple_scalar_add");
  int                n = 5;
  std::vector<float> a(n, 1.0f);
  std::vector<float> b(n, 0.0f);
  float              factor = 5.0f;

  // Bind wrong argument name on purpose (should log warning but not
  // crash/throw)
  EXPECT_NO_THROW({
    run.bind_buffer("wrong_name_a", a);
    run.bind_arguments(factor);
    run.bind_buffer("b", b);
  });
}
