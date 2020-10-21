#ifndef __GPU_UTILS_COMPACT_CUH__
#define __GPU_UTILS_COMPACT_CUH__

#include <thrust/unique.h>
#include <iostream>
#include <moderngpu/kernel_compact.hxx>
#include "CudaContext.cuh"
#include "DeviceArray.cuh"
#include "MGPUContext.cuh"
#include "Reduce.cuh"
#include "ThrustContext.cuh"

namespace GpuUtils {
namespace Compact {
// =========================== Compact =========================
// The most general Compact API using cub with known num_selected_out and
// iterator types
template <typename InputIterator, typename FlagIterator,
          typename OutputIterator>
void CubCompact(InputIterator input, int input_count, int num_selected_out,
                FlagIterator bitmaps, OutputIterator output,
                CudaContext* context) {
  int* d_num_selected_out = (int*)context->Malloc(sizeof(int));
  void* d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  assert(input_count >= 0);
  CUDA_ERROR(cub::DeviceSelect::Flagged(
      d_temp_storage, temp_storage_bytes, input, bitmaps, output,
      d_num_selected_out, input_count, context->Stream()));
  d_temp_storage = context->Malloc(temp_storage_bytes);
  CUDA_ERROR(cub::DeviceSelect::Flagged(
      d_temp_storage, temp_storage_bytes, input, bitmaps, output,
      d_num_selected_out, input_count, context->Stream()));
  context->Free(d_temp_storage, temp_storage_bytes);
  context->Free(d_num_selected_out, sizeof(int));
}

// A general API using cub, num_selected_out is unknown and output is not
// allocated.
template <typename InputIterator, typename FlagIterator, typename DataType>
void CubCompact(InputIterator input, int input_count, FlagIterator bitmaps,
                DataType*& output, int& num_selected_out,
                CudaContext* context) {
  int* d_num_selected_out = (int*)context->Malloc(sizeof(int));
  GpuUtils::Reduce::Sum(bitmaps, d_num_selected_out, input_count, context);
  CUDA_ERROR(cudaMemcpy(&num_selected_out, d_num_selected_out, sizeof(int),
                        cudaMemcpyDeviceToHost));
  context->Free(d_num_selected_out, sizeof(int));

  output = (DataType*)context->Malloc(sizeof(DataType) * num_selected_out);
  CubCompact(input, input_count, num_selected_out, bitmaps, output, context);
}

// ===================== entry ======================

// output_count is unknown and written on return
template <typename InputIterator, typename FlagIterator, typename DataType>
void Compact(InputIterator input, int input_count, FlagIterator bitmaps,
             DataType*& output, int& output_count, CudaContext* context) {
  CubCompact(input, input_count, bitmaps, output, output_count, context);
}

// num_selected_out is known
template <typename InputIterator, typename FlagIterator, typename DataType>
void Compact(InputIterator input, int input_count, int num_selected_out,
             FlagIterator bitmaps, DataType*& output, CudaContext* context) {
  if (!output)
    output = (DataType*)context->Malloc(sizeof(DataType) * num_selected_out);
  CubCompact(input, input_count, num_selected_out, bitmaps, output, context);
}

//////// with DeviceArray as the parameter
// output is of DeviceArray
template <typename InputIterator, typename FlagIterator, typename DataType>
void Compact(InputIterator input, int input_count, FlagIterator bitmaps,
             DeviceArray<DataType>*& output, int& output_count,
             CudaContext* context) {
  DataType* output_array = NULL;
  Compact(input, input_count, bitmaps, output_array, output_count, context);
  assert(output == NULL);
  output = new DeviceArray<DataType>(output_array, output_count, context, true);
}
template <typename InputIterator, typename FlagIterator, typename DataType>
void Compact(InputIterator input, int input_count, int num_selected_out,
             FlagIterator bitmaps, DeviceArray<DataType>*& output,
             CudaContext* context) {
  DataType* output_array = NULL;
  Compact(input, input_count, num_selected_out, bitmaps, output_array, context);
  assert(output == NULL);
  output =
      new DeviceArray<DataType>(output_array, num_selected_out, context, true);
}

// input, output are of DeviceArray
template <typename DataType, typename FlagIterator>
void Compact(DeviceArray<DataType>* input, int input_count,
             FlagIterator bitmaps, DeviceArray<DataType>*& output,
             int& output_count, CudaContext* context) {
  Compact(input->GetArray(), input_count, bitmaps, output, output_count,
          context);
}
template <typename DataType, typename FlagIterator>
void Compact(DeviceArray<DataType>* input, int input_count,
             int num_selected_out, FlagIterator bitmaps,
             DeviceArray<DataType>*& output, CudaContext* context) {
  Compact(input->GetArray(), input_count, num_selected_out, bitmaps, output,
          context);
}

// with DeviceArray as the parameter
// directly write to the input array and release the memory of the orignial
// array
template <typename DataType, typename FlagIterator>
void Compact(DeviceArray<DataType>*& input, int input_count,
             FlagIterator bitmaps, int& output_count, CudaContext* context) {
  DeviceArray<DataType>* compact_output = NULL;
  Compact(input, input_count, bitmaps, compact_output, output_count, context);
  std::swap(input, compact_output);
  delete compact_output;
  compact_output = NULL;
}
template <typename DataType, typename FlagIterator>
void Compact(DeviceArray<DataType>*& input, int input_count,
             int num_selected_out, FlagIterator bitmaps, CudaContext* context) {
  DeviceArray<DataType>* compact_output = NULL;
  Compact(input, input_count, num_selected_out, bitmaps, compact_output,
          context);
  std::swap(input, compact_output);
  delete compact_output;
  compact_output = NULL;
}

// ========================== Unique ===============================
// Requirement: data is sorted
// Return: the unique counts returned
template <typename DataType>
size_t Unique(DataType* data, size_t count, CudaContext* context) {
  thrust::device_ptr<DataType> data_ptr(data);
  return thrust::unique(CustomPolicy(context), data_ptr, data_ptr + count) -
         data_ptr;
}
// Requirement: data is sorted.
// f(DataType left, DataType right)
// Return: the unique counts returned
template <typename DataType, typename Func>
size_t Unique(DataType* data, size_t count, Func f, CudaContext* context) {
  thrust::device_ptr<DataType> data_ptr(data);
  return thrust::unique(CustomPolicy(context), data_ptr, data_ptr + count, f) -
         data_ptr;
}

}  // namespace Compact
}  // namespace GpuUtils

#endif
