#ifndef __GPU_UTILS_THRUST_CONTEXT_CUH__
#define __GPU_UTILS_THRUST_CONTEXT_CUH__

#include <thrust/execution_policy.h>
#include <thrust/pair.h>
#include "CudaContext.cuh"

// Specify my own execution policy so as to intercept the
// temporary memory allocation for Thrust algorithm execution.
// We use UnTrackMalloc and UnTrackFree interface to manage memory.
// This is because return_temporary_buffer cannot have the memory size
// in the argument, and thus we cannot track the memory statistics in
// CudaContext.
// This can cause potentially incorrect memory statistics when using
// CustomPolicy.
// This may not be an issue because:
// 1. The internal memory allocation is small, so the incorrect memory
// statistics may not cause the memory overflow.
// 2. Using CnmemCudaContext can allow potential memory overflow.
struct CustomPolicy : thrust::device_execution_policy<CustomPolicy> {
  CustomPolicy(CudaContext *ctx)
      : context_(ctx), thrust::device_execution_policy<CustomPolicy>() {}
  CudaContext *context_;
};

template <typename T>
thrust::pair<thrust::pointer<T, CustomPolicy>, std::ptrdiff_t>
get_temporary_buffer(CustomPolicy my_policy, std::ptrdiff_t n) {
  thrust::pointer<T, CustomPolicy> result(
      static_cast<T *>(my_policy.context_->UnTrackMalloc(n * sizeof(T))));
  // return the pointer and the number of elements allocated
  return thrust::make_pair(result, n);
}

template <typename Pointer>
void return_temporary_buffer(CustomPolicy my_policy, Pointer p) {
  my_policy.context_->UnTrackFree(thrust::device_pointer_cast(p.get()).get());
}
#endif
