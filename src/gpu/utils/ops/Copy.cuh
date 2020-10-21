#ifndef __GPU_UTILS_COPY_CUH__
#define __GPU_UTILS_COPY_CUH__

#include <thrust/device_ptr.h>
#include <thrust/gather.h>
#include <vector>
#include "CudaContext.cuh"
#include "Meta.h"
#include "ThrustContext.cuh"
#include "Transform.cuh"

namespace GpuUtils {
namespace Copy {

// =========== Gather ===========
//[indices, indices+indices_count) and [output, output+indices_count) cannot
// overlap
template <typename IndexType, typename DataType>
void Gather(IndexType *indices, size_t indices_count, DataType *input,
            DataType *output, CudaContext *context) {
  thrust::device_ptr<DataType> input_ptr(input);
  thrust::device_ptr<DataType> output_ptr(output);
  thrust::device_ptr<IndexType> indices_ptr(indices);
  thrust::gather(CustomPolicy(context), indices_ptr,
                 indices_ptr + indices_count, input_ptr, output_ptr);
}

}  // namespace Copy
}  // namespace GpuUtils
#endif
