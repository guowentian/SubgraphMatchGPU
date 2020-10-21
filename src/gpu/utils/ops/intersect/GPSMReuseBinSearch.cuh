#ifndef __GPU_SET_INTERSECT_GPSM_REUSE_BINARY_SEARCH_CUH__
#define __GPU_SET_INTERSECT_GPSM_REUSE_BINARY_SEARCH_CUH__

#include "GPSMBinSearch.cuh"

namespace GpuUtils {
namespace Intersect {
namespace GPSMReuseDetail {
template <bool kCacheRequired>
struct GPSMReuseCountFunctor {
  template <typename WarpPrepareFunctor, typename GenCandidatesFunctor,
            typename VerifyFunctor, typename CountFunctor, typename CountType>
  DEVICE void operator()(size_t path_id,
                         WarpPrepareFunctor warp_prepare_functor,
                         GenCandidatesFunctor gen_candidates_functor,
                         VerifyFunctor verify_functor,
                         CountFunctor count_functor, CountType* output_count,
                         CountType* intersect_output_count) {
    __shared__ uintV M[WARPS_PER_BLOCK][kMaxQueryVerticesNum];
    __shared__ bool shm_pivot_from_cache[WARPS_PER_BLOCK];
    __shared__ uintV shm_pivot_index[WARPS_PER_BLOCK];
    __shared__ uintV shm_pivot_level[WARPS_PER_BLOCK];

    typedef cub::WarpReduce<size_t> WarpReduce;
    __shared__
        typename WarpReduce::TempStorage warp_reduce_storage[WARPS_PER_BLOCK];

    int lane_id = threadIdx.x % THREADS_PER_WARP;
    int warp_id = threadIdx.x / THREADS_PER_WARP;

    if (lane_id == 0) {
      warp_prepare_functor(M[warp_id], path_id, shm_pivot_from_cache[warp_id],
                           shm_pivot_index[warp_id], shm_pivot_level[warp_id]);
    }

    WarpSync();

    uintV* candidates = NULL;
    size_t candidates_count = 0;

    gen_candidates_functor(M[warp_id], path_id, shm_pivot_from_cache[warp_id],
                           shm_pivot_index[warp_id], shm_pivot_level[warp_id],
                           candidates, candidates_count);

    CountType thread_valid_count = 0;
    CountType thread_intersect_valid_count = 0;
    CountType move_index = lane_id;
    while (move_index < candidates_count) {
      auto candidate = candidates[move_index];
      bool valid, intersect_valid;
      verify_functor(M[warp_id], path_id, shm_pivot_from_cache[warp_id],
                     shm_pivot_index[warp_id], shm_pivot_level[warp_id],
                     candidate, intersect_valid, valid);
      thread_intersect_valid_count += (intersect_valid ? 1 : 0);
      if (valid) {
        thread_valid_count += count_functor(path_id, candidate);
      }
      move_index += THREADS_PER_WARP;
    }

    CountType warp_aggregates =
        WarpReduce(warp_reduce_storage[warp_id]).Sum(thread_valid_count);
    if (lane_id == 0) {
      output_count[path_id] = warp_aggregates;
    }

    if (kCacheRequired) {
      CountType warp_aggregates = WarpReduce(warp_reduce_storage[warp_id])
                                      .Sum(thread_intersect_valid_count);
      if (lane_id == 0) {
        intersect_output_count[path_id] = warp_aggregates;
      }
    }
  }
};

template <bool kCacheRequired>
struct GPSMReuseWriteFunctor {
  template <typename WarpPrepareFunctor, typename GenCandidatesFunctor,
            typename VerifyFunctor, typename CountType>
  DEVICE void operator()(size_t path_id,
                         WarpPrepareFunctor warp_prepare_functor,
                         GenCandidatesFunctor gen_candidates_functor,
                         VerifyFunctor verify_functor, CountType* output_offset,
                         uintV* output, CountType* intersect_output_offset,
                         uintV* intersect_output) {
    __shared__ uintV M[WARPS_PER_BLOCK][kMaxQueryVerticesNum];
    __shared__ bool shm_pivot_from_cache[WARPS_PER_BLOCK];
    __shared__ uintV shm_pivot_index[WARPS_PER_BLOCK];
    __shared__ uintV shm_pivot_level[WARPS_PER_BLOCK];
    __shared__ size_t shm_warp_offsets[WARPS_PER_BLOCK];
    __shared__ size_t shm_warp_intersect_write_offsets[WARPS_PER_BLOCK];

    typedef cub::WarpScan<size_t> WarpScan;
    __shared__
        typename WarpScan::TempStorage warp_scan_storage[WARPS_PER_BLOCK];

    int lane_id = threadIdx.x % THREADS_PER_WARP;
    int warp_id = threadIdx.x / THREADS_PER_WARP;

    if (lane_id == 0) {
      warp_prepare_functor(M[warp_id], path_id, shm_pivot_from_cache[warp_id],
                           shm_pivot_index[warp_id], shm_pivot_level[warp_id]);
      shm_warp_offsets[warp_id] = 0;
      shm_warp_intersect_write_offsets[warp_id] = 0;
    }

    WarpSync();

    uintV* candidates = NULL;
    CountType candidates_count = 0;

    gen_candidates_functor(M[warp_id], path_id, shm_pivot_from_cache[warp_id],
                           shm_pivot_index[warp_id], shm_pivot_level[warp_id],
                           candidates, candidates_count);

    uintV* warp_output = output + output_offset[path_id];
    uintV* warp_intersect_output =
        kCacheRequired ? intersect_output + intersect_output_offset[path_id]
                       : NULL;

    CountType move_index = lane_id;
    while (WarpSyncOr(move_index < candidates_count)) {
      bool valid = false, intersect_valid = false;
      uintV candidate = 0;
      if (move_index < candidates_count) {
        candidate = candidates[move_index];
        verify_functor(M[warp_id], path_id, shm_pivot_from_cache[warp_id],
                       shm_pivot_index[warp_id], shm_pivot_level[warp_id],
                       candidate, intersect_valid, valid);
      }

      size_t thread_offset, warp_aggregates;
      WarpScan(warp_scan_storage[warp_id])
          .ExclusiveSum(valid ? 1 : 0, thread_offset, warp_aggregates);
      if (valid) {
        warp_output[shm_warp_offsets[warp_id] + thread_offset] = candidate;
      }
      if (lane_id == 0) {
        shm_warp_offsets[warp_id] += warp_aggregates;
      }

      if (kCacheRequired) {
        WarpScan(warp_scan_storage[warp_id])
            .ExclusiveSum(intersect_valid ? 1 : 0, thread_offset,
                          warp_aggregates);
        if (intersect_valid) {
          warp_intersect_output[shm_warp_intersect_write_offsets[warp_id] +
                                thread_offset] = candidate;
        }
        if (lane_id == 0) {
          shm_warp_intersect_write_offsets[warp_id] += warp_aggregates;
        }
      }

      move_index += THREADS_PER_WARP;
    }
  }
};
}  // namespace GPSMReuseDetail

// The general interface for GpsmReuseBinSearch
// kCacheRequiared: whether to store the result for intersect_output_row_ptrs
// and intersect_output_cols warp_prepare_functor: prepare shared data structure
// for the same warp gen_candidates_functor: prepare candidates and
// candidates_count verify_functor: to check the validity for a candidate
template <bool kCacheRequired, typename WarpPrepareFunctor,
          typename GenCandidatesFunctor, typename VerifyFunctor,
          typename CountType>
void GpsmReuseBinSearch(WarpPrepareFunctor warp_prepare_functor,
                        GenCandidatesFunctor gen_candidates_functor,
                        VerifyFunctor verify_functor, size_t path_num,
                        CountType*& output_row_ptrs, uintV*& output_cols,
                        CountType*& intersect_output_row_ptrs,
                        uintV*& intersect_output_cols, CudaContext* context) {
  CountType* output_count =
      (CountType*)context->Malloc(sizeof(CountType) * path_num);
  CountType* intersect_output_count =
      kCacheRequired ? (CountType*)context->Malloc(sizeof(CountType) * path_num)
                     : NULL;

  auto count_functor = [=] DEVICE(size_t path_id, uintV candidate) {
    return 1;
  };
  GpuUtils::Transform::WarpTransform(
      GPSMReuseDetail::GPSMReuseCountFunctor<kCacheRequired>(), path_num,
      context, warp_prepare_functor, gen_candidates_functor, verify_functor,
      count_functor, output_count, intersect_output_count);

  output_row_ptrs =
      (CountType*)context->Malloc(sizeof(CountType) * (path_num + 1));
  GpuUtils::Scan::ExclusiveSum(output_count, path_num, output_row_ptrs,
                               output_row_ptrs + path_num, context);
  context->Free(output_count, sizeof(CountType) * path_num);
  output_count = NULL;

  CountType total_output_count = GetD(output_row_ptrs + path_num);
  output_cols = (uintV*)context->Malloc(sizeof(uintV) * total_output_count);

  intersect_output_row_ptrs = NULL;
  if (kCacheRequired) {
    intersect_output_row_ptrs =
        (CountType*)context->Malloc(sizeof(CountType) * (path_num + 1));
    GpuUtils::Scan::ExclusiveSum(intersect_output_count, path_num,
                                 intersect_output_row_ptrs,
                                 intersect_output_row_ptrs + path_num, context);
    context->Free(intersect_output_count, sizeof(CountType) * path_num);
    intersect_output_count = NULL;

    CountType total_intersect_output_count =
        GetD(intersect_output_row_ptrs + path_num);
    intersect_output_cols =
        (uintV*)context->Malloc(sizeof(uintV) * total_intersect_output_count);
  }

  GpuUtils::Transform::WarpTransform(
      GPSMReuseDetail::GPSMReuseWriteFunctor<kCacheRequired>(), path_num,
      context, warp_prepare_functor, gen_candidates_functor, verify_functor,
      output_row_ptrs, output_cols, intersect_output_row_ptrs,
      intersect_output_cols);
}

// With kCacheRequired as function parameter
template <typename WarpPrepareFunctor, typename GenCandidatesFunctor,
          typename VerifyFunctor, typename CountType>
void GpsmReuseBinSearch(WarpPrepareFunctor warp_prepare_functor,
                        GenCandidatesFunctor gen_candidates_functor,
                        VerifyFunctor verify_functor, size_t path_num,
                        CountType*& output_row_ptrs, uintV*& output_cols,
                        CountType*& intersect_output_row_ptrs,
                        uintV*& intersect_output_cols, CudaContext* context,
                        const bool kCacheRequired) {
  if (kCacheRequired) {
    GpsmReuseBinSearch<true>(warp_prepare_functor, gen_candidates_functor,
                             verify_functor, path_num, output_row_ptrs,
                             output_cols, intersect_output_row_ptrs,
                             intersect_output_cols, context);
  } else {
    GpsmReuseBinSearch<false>(warp_prepare_functor, gen_candidates_functor,
                              verify_functor, path_num, output_row_ptrs,
                              output_cols, intersect_output_row_ptrs,
                              intersect_output_cols, context);
  }
}

// With kCacheRequired as function parameter
// With DeviceArray as the parameters
template <typename WarpPrepareFunctor, typename GenCandidatesFunctor,
          typename VerifyFunctor, typename CountType>
void GpsmReuseBinSearch(WarpPrepareFunctor warp_prepare_functor,
                        GenCandidatesFunctor gen_candidates_functor,
                        VerifyFunctor verify_functor, size_t path_num,
                        DeviceArray<CountType>*& output_row_ptrs,
                        DeviceArray<uintV>*& output_cols,
                        DeviceArray<CountType>*& intersect_output_row_ptrs,
                        DeviceArray<uintV>*& intersect_output_cols,
                        CudaContext* context, const bool kCacheRequired) {
  CountType *output_row_ptrs_data = NULL,
            *intersect_output_row_ptrs_data = NULL;
  uintV *output_cols_data = NULL, *intersect_output_cols_data = NULL;

  GpsmReuseBinSearch(warp_prepare_functor, gen_candidates_functor,
                     verify_functor, path_num, output_row_ptrs_data,
                     output_cols_data, intersect_output_row_ptrs_data,
                     intersect_output_cols_data, context, kCacheRequired);

  output_row_ptrs = new DeviceArray<CountType>(output_row_ptrs_data,
                                               path_num + 1, context, true);
  CountType total_output_count = GetD(output_row_ptrs_data + path_num);
  output_cols = new DeviceArray<uintV>(output_cols_data, total_output_count,
                                       context, true);

  if (kCacheRequired) {
    intersect_output_row_ptrs = new DeviceArray<CountType>(
        intersect_output_row_ptrs_data, path_num + 1, context, true);
    CountType total_intersect_output_count =
        GetD(intersect_output_row_ptrs_data + path_num);
    intersect_output_cols =
        new DeviceArray<uintV>(intersect_output_cols_data,
                               total_intersect_output_count, context, true);
  }
}

}  // namespace Intersect
}  // namespace GpuUtils

#endif