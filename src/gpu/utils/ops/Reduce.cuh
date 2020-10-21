#ifndef __GPU_UTILS_REDUCE_CUH__
#define __GPU_UTILS_REDUCE_CUH__

#include <cub/cub.cuh>
#include <moderngpu/kernel_reduce.hxx>
#include "CudaContext.cuh"
#include "MGPUContext.cuh"

namespace GpuUtils {
namespace Reduce {

// output is a pre-allocated device pointer which stores the result of reduce
template <typename InputIt, typename OutputIt>
void Sum(InputIt input, OutputIt output, int count, CudaContext* context) {
  void* d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, input, output,
                         count, context->Stream());
  d_temp_storage = context->Malloc(temp_storage_bytes);
  cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, input, output,
                         count, context->Stream());
  context->Free(d_temp_storage, temp_storage_bytes);
}

// func(int index) -> ValueType
template <typename launch_arg_t = mgpu::empty_t, typename Func,
          typename OutputIt>
void TransformReduce(Func func, int count, OutputIt reduction,
                     CudaContext* context) {
  typedef
      typename mgpu::conditional_typedef_t<launch_arg_t, MGPULaunchBox>::type_t
          LaunchBox;
  typedef typename std::iterator_traits<OutputIt>::value_type ValueType;
  mgpu::transform_reduce(func, count, reduction, mgpu::plus_t<ValueType>(),
                         *context);
}

}  // namespace Reduce
}  // namespace GpuUtils

#endif
