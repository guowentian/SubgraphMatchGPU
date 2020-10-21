#ifndef __SCAN_CUH__
#define __SCAN_CUH__

#include <cub/cub.cuh>
#include <iterator>
#include <moderngpu/kernel_scan.hxx>
#include "MGPUContext.cuh"

namespace GpuUtils {
namespace Scan {

// from cub
template <typename InputIt, typename OutputIt>
void CubExclusiveSum(InputIt children_count, int parent_count,
                     OutputIt children_offset, CudaContext *context) {
  cudaStream_t stream = context->Stream();
  void *d_temp_storage = NULL;
  size_t d_temp_storage_bytes = 0;
  cub::DeviceScan::ExclusiveSum(d_temp_storage, d_temp_storage_bytes,
                                children_count, children_offset, parent_count,
                                stream);
  d_temp_storage = context->Malloc(d_temp_storage_bytes);
  cub::DeviceScan::ExclusiveSum(d_temp_storage, d_temp_storage_bytes,
                                children_count, children_offset, parent_count,
                                stream);
  context->Free(d_temp_storage, d_temp_storage_bytes);
}

template <typename InputIt, typename OutputIt>
void CubExclusiveSum(InputIt children_count, int parent_count,
                     OutputIt children_offset, OutputIt reduction,
                     CudaContext *context) {
  CubExclusiveSum(children_count, parent_count, children_offset, context);

  typedef typename std::iterator_traits<OutputIt>::value_type DataType;
  DataType total_children_count = 0;
  DToH(&total_children_count, children_offset + parent_count - 1, 1);
  DataType last_count = 0;
  DToH(&last_count, children_count + parent_count - 1, 1);
  total_children_count += last_count;
  HToD(reduction, &total_children_count, 1);
}
// from mgpu
template <typename InputIt, typename OutputIt>
void MgpuExclusiveSum(InputIt input_it, int count, OutputIt output_it,
                      OutputIt reduction_it, CudaContext *context) {
  typedef typename std::iterator_traits<OutputIt>::value_type DataType;
  mgpu::scan<mgpu::scan_type_exc, MGPULaunchBox>(input_it, count, output_it,
                                                 mgpu::plus_t<DataType>(),
                                                 reduction_it, *context);
}

template <typename InputIt, typename OutputIt>
void MgpuExclusiveSum(InputIt input_it, int count, OutputIt output_it,
                      CudaContext *context) {
  typedef typename std::iterator_traits<OutputIt>::value_type DataType;
  mgpu::scan<mgpu::scan_type_exc, MGPULaunchBox>(input_it, count, output_it,
                                                 *context);
}

// ===================== entry ===================
// func: index -> count
template <typename Func, typename OutputIterator, typename ReductionType>
void TransformScan(Func func, int count, OutputIterator output_it,
                   ReductionType reduction, CudaContext *context) {
  typedef typename std::iterator_traits<OutputIterator>::value_type DataType;
  mgpu::transform_scan<DataType, mgpu::scan_type_exc, MGPULaunchBox>(
      func, count, output_it, mgpu::plus_t<DataType>(), reduction, *context);
}

// note that ExclusiveSum only processes count elements, instead of count+1
template <typename InputIt, typename OutputIt>
void ExclusiveSum(InputIt input_it, int count, OutputIt output_it,
                  OutputIt reduction_it, CudaContext *context) {
  CubExclusiveSum(input_it, count, output_it, reduction_it, context);
}

template <typename InputIt, typename OutputIt>
void ExclusiveSum(InputIt input_it, int count, OutputIt output_it,
                  CudaContext *context) {
  CubExclusiveSum(input_it, count, output_it, context);
}

}  // namespace Scan
}  // namespace GpuUtils
#endif
