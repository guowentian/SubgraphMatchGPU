#ifndef __GPU_SET_INTERSECT_GPSM_REUSE_BINARY_SEARCH_COUNT_CUH__
#define __GPU_SET_INTERSECT_GPSM_REUSE_BINARY_SEARCH_COUNT_CUH__

#include "GPSMReuseBinSearch.cuh"
#include "Reduce.cuh"

namespace GpuUtils {
namespace Intersect {
// kReduce: if true, aggregate the count of each path and return on single
// result
// warp_prepare_functor: prepare shared data structure
// for the same warp.
// gen_candidates_functor: prepare candidates and
// candidates_count.
// verify_functor: to check the validity for a candidate
// count_functor: if candidate is valid, the increase count contributed
template <bool kReduce, typename WarpPrepareFunctor,
          typename GenCandidatesFunctor, typename VerifyFunctor,
          typename CountFunctor, typename CountType>
size_t GpsmReuseBinSearchCount(WarpPrepareFunctor warp_prepare_functor,
                               GenCandidatesFunctor gen_candidates_functor,
                               VerifyFunctor verify_functor, size_t path_num,
                               CountFunctor count_functor,
                               CountType*& output_offsets,
                               CudaContext* context) {
  DeviceArray<CountType> output_count(path_num, context);

  GpuUtils::Transform::WarpTransform(
      GPSMReuseDetail::GPSMReuseCountFunctor<kCacheRequired>(), path_num,
      context, warp_prepare_functor, gen_candidates_functor, verify_functor,
      count_functor, output_count.GetArray(), intersect_output_count);

  if (kReduce) {
    DeviceArray<size_t> d_total_count(1, context);
    GpuUtils::Reduce::Sum(output_count.GetArray(), d_total_count.GetArray(),
                          path_num, context);
    return GetD(d_total_count.GetArray());
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