#ifndef __GPU_UTILS_SORT_CUH__
#define __GPU_UTILS_SORT_CUH__

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include "ThrustContext.cuh"

namespace GpuUtils {
namespace Sort {
// the output is written to data
template <typename DataType>
void Sort(DataType* data, size_t count, CudaContext* context) {
  thrust::sort(CustomPolicy(context), thrust::device_ptr<DataType>(data),
               thrust::device_ptr<DataType>(data + count));
}

// the output is written to data
// f(DataType left, DataType right)
template <typename DataType, typename Func>
void Sort(DataType* data, size_t count, Func f, CudaContext* context) {
  thrust::sort(CustomPolicy(context), thrust::device_ptr<DataType>(data),
               thrust::device_ptr<DataType>(data + count), f);
}
}  // namespace Sort
}  // namespace GpuUtils

#endif
