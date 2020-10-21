#ifndef __GPU_SET_INTERSECT_GPSM_BINARY_SEARCH_CUH__
#define __GPU_SET_INTERSECT_GPSM_BINARY_SEARCH_CUH__

#include "CheckConstraintsCommon.cuh"
#include "CudaContext.cuh"
#include "DevPlan.cuh"
#include "Scan.cuh"
#include "Transform.cuh"

#include <cub/cub.cuh>

namespace GpuUtils {
namespace Intersect {
namespace GPSMDetail {
struct GPSMCountFunctor {
  // instance_gather_functor (path_id, M): construct the partial intance given a
  // path_id.
  // verify_functor (M, candidate, pivot_level): verify whether a
  // candidate is valid.
  // count_functor (path_id, candidate) -> size_t (count):
  // the increase count if a candidate is valid
  template <typename InstanceGatherFunctor, typename VerifyFunctor,
            typename CountFunctor, typename CountType>
  DEVICE void operator()(size_t path_id,
                         InstanceGatherFunctor instance_gather_functor,
                         VerifyFunctor verify_functor,
                         CountFunctor count_functor, CountType* output_count,
                         uintE* row_ptrs, uintV* cols, DevConnType* conn) {
    __shared__ uintV M[WARPS_PER_BLOCK][kMaxQueryVerticesNum];
    __shared__ uintV shm_pivot_level[WARPS_PER_BLOCK];

    typedef cub::WarpReduce<CountType> WarpReduce;
    __shared__
        typename WarpReduce::TempStorage warp_reduce_storage[WARPS_PER_BLOCK];

    int lane_id = threadIdx.x % THREADS_PER_WARP;
    int warp_id = threadIdx.x / THREADS_PER_WARP;

    // initialize warp-level shared data
    if (lane_id == 0) {
      instance_gather_functor(path_id, M[warp_id]);
      shm_pivot_level[warp_id] =
          ThreadChoosePivotLevel(*conn, M[warp_id], row_ptrs);
    }
    BlockSync();

    auto pivot_level = shm_pivot_level[warp_id];
    auto pivot_vertex = M[warp_id][pivot_level];
    auto candidates = cols + row_ptrs[pivot_vertex];
    CountType candidates_count =
        row_ptrs[pivot_vertex + 1] - row_ptrs[pivot_vertex];

    CountType thread_valid_count = 0;
    CountType move_index = lane_id;
    while (move_index < candidates_count) {
      auto candidate = candidates[move_index];
      bool valid = verify_functor(M[warp_id], candidate, pivot_level);
      if (valid) {
        thread_valid_count += count_functor(path_id, candidate);
      }
      move_index += THREADS_PER_WARP;
    }

    // warp reduce
    CountType warp_aggregates =
        WarpReduce(warp_reduce_storage[warp_id]).Sum(thread_valid_count);
    if (lane_id == 0) {
      output_count[path_id] = warp_aggregates;
    }
  }
};

struct GPSMWriteFunctor {
  // verify_functor (M, candidate, pivot_level)
  template <typename InstanceGatherFunctor, typename VerifyFunctor,
            typename CountType>
  DEVICE void operator()(size_t path_id,
                         InstanceGatherFunctor instance_gather_functor,
                         VerifyFunctor verify_functor, uintV* output,
                         CountType* output_offsets, uintE* row_ptrs,
                         uintV* cols, DevConnType* conn) {
    __shared__ uintV M[WARPS_PER_BLOCK][kMaxQueryVerticesNum];
    __shared__ uintV shm_pivot_level[WARPS_PER_BLOCK];

    // for WRITE, keep track of the current offset in the warp
    __shared__ CountType shm_warp_offsets[WARPS_PER_BLOCK];
    typedef cub::WarpScan<CountType> WarpScan;
    __shared__
        typename WarpScan::TempStorage warp_scan_storage[WARPS_PER_BLOCK];

    int lane_id = threadIdx.x % THREADS_PER_WARP;
    int warp_id = threadIdx.x / THREADS_PER_WARP;

    // initialize warp-level shared data
    if (lane_id == 0) {
      instance_gather_functor(path_id, M[warp_id]);
      shm_pivot_level[warp_id] =
          ThreadChoosePivotLevel(*conn, M[warp_id], row_ptrs);
      shm_warp_offsets[warp_id] = 0;
    }
    WarpSync();

    auto pivot_level = shm_pivot_level[warp_id];
    auto pivot_vertex = M[warp_id][pivot_level];
    auto candidates = cols + row_ptrs[pivot_vertex];
    CountType candidates_count =
        row_ptrs[pivot_vertex + 1] - row_ptrs[pivot_vertex];

    auto warp_output = output + output_offsets[path_id];
    CountType move_index = lane_id;
    while (WarpSyncOr(move_index < candidates_count)) {
      bool valid = false;
      uintV candidate;
      if (move_index < candidates_count) {
        candidate = candidates[move_index];
        valid = verify_functor(M[warp_id], candidate, pivot_level);
      }

      // WRITE
      // One warp scan in each iteration.
      // An alternative is use the two-step output scheme in the warp. We can
      // first count the valid count for each thread, and then warp scan to
      // obtain the offset for all the valid items for a thread. But this
      // method cannot guarantee the output is sorted. As each adjacent list
      // is small, we anticipate candidates_count is not large. So we don't
      // need too many warp scans in total.
      CountType thread_offset;
      CountType warp_aggregates;
      WarpScan(warp_scan_storage[warp_id])
          .ExclusiveSum(valid ? 1 : 0, thread_offset, warp_aggregates);
      if (valid) {
        warp_output[shm_warp_offsets[warp_id] + thread_offset] = candidate;
      }
      if (lane_id == 0) {
        shm_warp_offsets[warp_id] += warp_aggregates;
      }
      move_index += THREADS_PER_WARP;
    }
  }
};
}  // namespace GPSMDetail

// verify_functor (M, candidate, pivot_level)
template <bool kCheckCondition, bool kCheckDuplicates,
          typename InstanceGatherFunctor, typename CountType>
void GpsmBinSearch(InstanceGatherFunctor instance_gather_functor,
                   size_t path_num, uintE* row_ptrs, uintV* cols,
                   DevConnType* conn, DevCondArrayType* cond,
                   uintP* d_partition_ids, const uintV prime_edge_v0,
                   const uintV prime_edge_v1, CountType*& output_row_ptrs,
                   uintV*& output_cols, CudaContext* context) {
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
  auto count_functor = [=] DEVICE(size_t path_id, uintV candidate) {
    return 1;
  };

  CountType* output_count =
      (CountType*)context->Malloc(sizeof(CountType) * path_num);

  GpuUtils::Transform::WarpTransform(GPSMDetail::GPSMCountFunctor(), path_num,
                                     context, instance_gather_functor,
                                     verify_functor, count_functor,
                                     output_count, row_ptrs, cols, conn);

  output_row_ptrs =
      (CountType*)context->Malloc(sizeof(CountType) * (path_num + 1));
  GpuUtils::Scan::ExclusiveSum(output_count, path_num, output_row_ptrs,
                               output_row_ptrs + path_num, context);

  context->Free(output_count, sizeof(CountType) * path_num);
  output_count = NULL;

  CountType total_children_count;
  DToH(&total_children_count, output_row_ptrs + path_num, 1);
  output_cols = (uintV*)context->Malloc(sizeof(uintV) * total_children_count);

  GpuUtils::Transform::WarpTransform(GPSMDetail::GPSMWriteFunctor(), path_num,
                                     context, instance_gather_functor,
                                     verify_functor, output_cols,
                                     output_row_ptrs, row_ptrs, cols, conn);
}
}  // namespace Intersect
}  // namespace GpuUtils

#endif
