#ifndef __GPU_SET_INTERSECT_GPSM_BINARY_SEARCH_COUNT_CUH__
#define __GPU_SET_INTERSECT_GPSM_BINARY_SEARCH_COUNT_CUH__

#include "GPSMBinSearch.cuh"
#include "Reduce.cuh"

namespace GpuUtils {
namespace Intersect {
// count_functor (path_id, candidate) -> size_t (count)
template <bool kReduce, bool kCheckCondition, bool kCheckDuplicates,
          typename InstanceGatherFunctor, typename CountFunctor,
          typename CountType>
size_t GpsmBinSearchCount(InstanceGatherFunctor instance_gather_functor,
                          size_t path_num, uintE* row_ptrs, uintV* cols,
                          DevConnType* conn, DevCondArrayType* cond,
                          uintP* d_partition_ids, const uintV prime_edge_v0,
                          const uintV prime_edge_v1, CountFunctor count_functor,
                          CountType*& output_offsets, CudaContext* context) {
  auto verify_functor = [=] DEVICE(uintV * M, uintV candidate,
                                   uintV pivot_level) {
    bool valid = ThreadCheckConnectivity(*conn, M, candidate, pivot_level,
                                         row_ptrs, cols);
    if (kCheckCondition) {
      if (valid) {
        valid = ThreadCheckCondition(*cond, M, candidate);
      }
    }
    if (kCheckDuplicates) {
      if (valid) {
        valid = ThreadCheckDuplicate(*conn, M, candidate, d_partition_ids,
                                     prime_edge_v0, prime_edge_v1);
      }
    }
    return valid;
  };

  DeviceArray<CountType> output_count(path_num, context);
  CountType* output_count_array = output_count.GetArray();

  GpuUtils::Transform::WarpTransform(
      GPSMDetail::GPSMCountFunctor(), path_num, context,
      instance_gather_functor, verify_functor, count_functor,
      output_count.GetArray(), row_ptrs, cols, conn);

  if (kReduce) {
    DeviceArray<size_t> d_total_count(1, context);
    GpuUtils::Reduce::Sum(output_count.GetArray(), d_total_count.GetArray(),
                          path_num, context);

    size_t ret;
    DToH(&ret, d_total_count.GetArray(), 1);
    return ret;
  } else {
    output_offsets =
        (CountType*)context->Malloc(sizeof(CountType) * (path_num + 1));
    GpuUtils::Scan::ExclusiveSum(output_count.GetArray(), path_num,
                                 output_offsets, output_offsets + path_num,
                                 context);

    return 0;
  }
}

}  // namespace Intersect
}  // namespace GpuUtils

#endif