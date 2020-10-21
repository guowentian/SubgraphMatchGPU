#ifndef __TRANSFORM_CUH__
#define __TRANSFORM_CUH__

#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <moderngpu/transform.hxx>
#include "CudaContext.cuh"
#include "MGPUContext.cuh"
#include "Meta.h"
#include "ThrustContext.cuh"

namespace GpuUtils {
namespace Transform {

// ================== Transform ===================
// a general API that takes in a functor to perform any defined operations
template <typename Func, typename... Args>
void Transform(Func func, size_t count, CudaContext *context, Args... args) {
  mgpu::transform<MGPULaunchBox>(func, count, *context, args...);
}

// ================== apply ======================
// Based on the OperatorType, apply ADD, MAX, ASSIGNMENT etc..

// the same operation applied on an array
template <OperatorType op_type, typename DataType>
void Apply(DataType *array, size_t count, DataType val, CudaContext *context) {
  Transform(
      [=] DEVICE(size_t index) {
        switch (op_type) {
          case ADD:
            array[index] += val;
            break;
          case MULTIPLE:
            array[index] *= val;
            break;
          case ASSIGNMENT:
            array[index] = val;
            break;
          case MAX:
            array[index] = Max(array[index], val);
            break;
          case MIN:
            array[index] = Min(array[index], val);
            break;
          case MINUS:
            array[index] -= val;
            break;
          default:
            break;
        }
      },
      count, context);
}
template <typename DataType>
void Apply(DataType *array, size_t count, DataType val, OperatorType op_type,
           CudaContext *context) {
  switch (op_type) {
    case ADD:
      Apply<ADD>(array, count, val, context);
      break;
    case MULTIPLE:
      Apply<MULTIPLE>(array, count, val, context);
      break;
    case ASSIGNMENT:
      Apply<ASSIGNMENT>(array, count, val, context);
      break;
    case MAX:
      Apply<MAX>(array, count, val, context);
      break;
    case MIN:
      Apply<MIN>(array, count, val, context);
      break;
    case MINUS:
      Apply<MINUS>(array, count, val, context);
      break;
    default:
      break;
  }
}

// two arrays apply the operation
template <OperatorType op_type, typename DataType1, typename DataType2>
void Apply(DataType1 *array1, DataType2 *array2, size_t count,
           CudaContext *context) {
  Transform(
      [=] DEVICE(size_t index) {
        switch (op_type) {
          case ADD:
            array1[index] += array2[index];
            break;
          case MULTIPLE:
            array1[index] *= array2[index];
            break;
          case ASSIGNMENT:
            array1[index] = array2[index];
            break;
          case MAX:
            array1[index] =
                array1[index] > array2[index] ? array1[index] : array2[index];
            break;
          case MIN:
            array1[index] =
                array1[index] < array2[index] ? array1[index] : array2[index];
            break;
          case MINUS:
            array1[index] -= array2[index];
            break;
          default:
            break;
        }
      },
      count, context);
}

template <typename DataType>
void Apply(DataType *array1, DataType *array2, size_t count,
           OperatorType op_type, CudaContext *context) {
  switch (op_type) {
    case ADD:
      Apply<ADD>(array1, array2, count, context);
      break;
    case MULTIPLE:
      Apply<MULTIPLE>(array1, array2, count, context);
      break;
    case ASSIGNMENT:
      Apply<ASSIGNMENT>(array1, array2, count, context);
      break;
    case MAX:
      Apply<MAX>(array1, array2, count, context);
      break;
    case MIN:
      Apply<MIN>(array1, array2, count, context);
      break;
    case MINUS:
      Apply<MINUS>(array1, array2, count, context);
      break;
    default:
      break;
  }
}

// ================== sequence ==================
template <typename DataType>
void Sequence(DataType *array, size_t count, DataType init,
              CudaContext *context) {
  thrust::device_ptr<DataType> array_ptr(array);
  thrust::sequence(CustomPolicy(context), array_ptr, array_ptr + count, init);
}

// ================= GetCountFromOffset =============
template <typename IndexType, typename OffsetType, typename CountType>
void GetCountFromOffset(IndexType *indices, size_t indices_count,
                        OffsetType *row_ptrs, CountType *count,
                        CudaContext *context) {
  GpuUtils::Transform::Transform(
      [=] DEVICE(int index) {
        IndexType vertex_id = indices[index];
        count[index] = row_ptrs[vertex_id + 1] - row_ptrs[vertex_id];
      },
      indices_count, context);
}
template <typename IndexType, typename OffsetType, typename CountType>
void GetCountFromOffset(IndexType indices_count, OffsetType *row_ptrs,
                        CountType *count, CudaContext *context) {
  GpuUtils::Transform::Transform(
      [=] DEVICE(int index) {
        count[index] = row_ptrs[index + 1] - row_ptrs[index];
      },
      indices_count, context);
}

// ==================== Warp Transform =====================
template <typename Func, typename... Args>
__global__ void WarpTransformKernel(Func func, size_t count, Args... args) {
  size_t pid =
      ((size_t)blockIdx.x * blockDim.x + threadIdx.x) / THREADS_PER_WARP;
  if (pid < count) {
    func(pid, args...);
  }
}
// func(path_id, args...)
template <typename Func, typename... Args>
void WarpTransform(Func func, size_t count, CudaContext *context,
                   Args... args) {
  const int kWarpNumPerCTA = THREADS_PER_BLOCK / THREADS_PER_WARP;
  const int num_ctas = (count + kWarpNumPerCTA - 1) / kWarpNumPerCTA;
  WarpTransformKernel<<<num_ctas, THREADS_PER_BLOCK, 0, context->Stream()>>>(
      func, count, args...);
}

}  // namespace Transform
}  // namespace GpuUtils
#endif
