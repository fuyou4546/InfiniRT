#include <infini/rt.h>
#include INFINI_RT_TEST_DRIVER_HEADER
#include INFINI_RT_TEST_RUNTIME_HEADER

#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

#include "test_helper.h"

#if INFINI_RT_TEST_HAS_TRITON_KERNEL
constexpr const char kTritonKernelName[] = "empty_kernel";
constexpr unsigned int kTritonKernelBlockSize = 128;
#endif

namespace {

using Driver = infini::rt::driver::Driver<INFINI_RT_TEST_DEVICE_TYPE>;
using Runtime = infini::rt::runtime::Runtime<INFINI_RT_TEST_DEVICE_TYPE>;

bool SelectDevice() {
  int device_count = 0;
  if (Runtime::GetDeviceCount(&device_count) != Runtime::kSuccess ||
      device_count <= 0) {
    std::cout << INFINI_RT_TEST_BACKEND_NAME
              << " driver skipped: no available device." << std::endl;
    return false;
  }
  if (Runtime::SetDevice(0) != Runtime::kSuccess) {
    std::cout << INFINI_RT_TEST_BACKEND_NAME
              << " driver skipped: device 0 is not available." << std::endl;
    return false;
  }
  return true;
}

void TestDriverFunctionPointers(infini::rt::test::TestContext* context) {
  context->Expect(Driver::ModuleLoadData != nullptr, INFINI_RT_TEST_BACKEND_NAME
                  " driver ModuleLoadData should be non-null.");
  context->Expect(Driver::ModuleGetFunction != nullptr,
                  INFINI_RT_TEST_BACKEND_NAME
                  " driver ModuleGetFunction should be non-null.");
  context->Expect(Driver::ModuleUnload != nullptr, INFINI_RT_TEST_BACKEND_NAME
                  " driver ModuleUnload should be non-null.");
  context->Expect(Driver::LaunchKernel != nullptr, INFINI_RT_TEST_BACKEND_NAME
                  " driver LaunchKernel should be non-null.");
  context->Expect(Driver::FuncGetAttribute != nullptr,
                  INFINI_RT_TEST_BACKEND_NAME
                  " driver FuncGetAttribute should be non-null.");
  context->Expect(Driver::FuncSetCacheConfig != nullptr,
                  INFINI_RT_TEST_BACKEND_NAME
                  " driver FuncSetCacheConfig should be non-null.");
  context->Expect(Driver::FuncSetAttribute != nullptr,
                  INFINI_RT_TEST_BACKEND_NAME
                  " driver FuncSetAttribute should be non-null.");
}

void TestDriverAttributeConstants(infini::rt::test::TestContext* context) {
  int shared_size = static_cast<int>(Driver::kFuncAttributeSharedSizeBytes);
  context->Expect(shared_size >= 0, INFINI_RT_TEST_BACKEND_NAME
                  " driver kFuncAttributeSharedSizeBytes should be non-negative.");
  int cache_config = static_cast<int>(Driver::kFuncCachePreferShared);
  context->Expect(cache_config >= 0, INFINI_RT_TEST_BACKEND_NAME
                  " driver kFuncCachePreferShared should be non-negative.");
  int max_dynamic = static_cast<int>(Driver::kFuncAttributeMaxDynamicSharedSizeBytes);
  context->Expect(max_dynamic >= 0, INFINI_RT_TEST_BACKEND_NAME
                  " driver kFuncAttributeMaxDynamicSharedSizeBytes should be non-negative.");
}

void ExpectDriverSuccess(infini::rt::test::TestContext* context,
                         Driver::Result status, const char* message) {
  context->Expect(status == Driver::kSuccess, message);
}

void ExpectRuntimeSuccess(infini::rt::test::TestContext* context,
                          typename Runtime::Error status, const char* message) {
  context->Expect(status == Runtime::kSuccess, message);
}

void TestDriverKernelLifecycle(infini::rt::test::TestContext* context) {
#if INFINI_RT_TEST_HAS_TRITON_KERNEL
  std::ifstream file(INFINI_RT_TEST_TRITON_KERNEL_FILE, std::ios::binary);
  if (!file) {
    std::cout << INFINI_RT_TEST_BACKEND_NAME
              << " driver kernel-lifecycle test skipped: no kernel image "
              << INFINI_RT_TEST_TRITON_KERNEL_FILE << std::endl;
    return;
  }
  std::vector<char> image((std::istreambuf_iterator<char>(file)),
                          std::istreambuf_iterator<char>());
  image.push_back('\0');

  Driver::Module module;
  ExpectDriverSuccess(context,
                      Driver::ModuleLoadData(&module, image.data()),
                      INFINI_RT_TEST_BACKEND_NAME
                      " driver should load a Triton module.");

  Driver::Function function;
  ExpectDriverSuccess(context,
                      Driver::ModuleGetFunction(&function, module,
                                                kTritonKernelName),
                      INFINI_RT_TEST_BACKEND_NAME
                      " driver should resolve the kernel function.");

  int shared_size = 0;
  ExpectDriverSuccess(
      context,
      Driver::FuncGetAttribute(&shared_size,
                               Driver::kFuncAttributeSharedSizeBytes, function),
      INFINI_RT_TEST_BACKEND_NAME
      " driver should query the function shared size attribute.");
  context->Expect(shared_size >= 0, INFINI_RT_TEST_BACKEND_NAME
                  " driver function shared size should be non-negative.");

  ExpectDriverSuccess(
      context,
      Driver::FuncSetCacheConfig(function, Driver::kFuncCachePreferShared),
      INFINI_RT_TEST_BACKEND_NAME
      " driver should set the function cache config.");

  ExpectDriverSuccess(
      context,
      Driver::FuncSetAttribute(
          function, Driver::kFuncAttributeMaxDynamicSharedSizeBytes, 0),
      INFINI_RT_TEST_BACKEND_NAME
      " driver should set the function dynamic shared size attribute.");

  std::uint64_t scratch = 0;
  void* params[2] = {&scratch, &scratch};
  ExpectDriverSuccess(
      context,
      Driver::LaunchKernel(function, 1, 1, 1, kTritonKernelBlockSize, 1, 1, 0,
                           nullptr, params, nullptr),
      INFINI_RT_TEST_BACKEND_NAME
      " driver should launch the kernel.");
  ExpectRuntimeSuccess(context, Runtime::DeviceSynchronize(),
                       INFINI_RT_TEST_BACKEND_NAME
                       " driver should synchronize after launch.");

  ExpectDriverSuccess(context, Driver::ModuleUnload(module),
                      INFINI_RT_TEST_BACKEND_NAME
                      " driver should unload the module.");
#endif
}

}  // namespace

int main() {
  infini::rt::test::TestContext context;
  if (!SelectDevice()) return context.ExitCode();
  TestDriverFunctionPointers(&context);
  TestDriverAttributeConstants(&context);
  TestDriverKernelLifecycle(&context);
  return context.ExitCode();
}
