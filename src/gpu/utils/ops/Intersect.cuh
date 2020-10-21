#ifndef __GPU_UTILS_OPS_SET_INTERSECTION_CUH__
#define __GPU_UTILS_OPS_SET_INTERSECTION_CUH__

#include "GPSMBinSearch.cuh"
#include "GPSMBinSearchCount.cuh"
#include "Meta.h"

namespace GpuUtils {
namespace Intersect {

enum ProcessMethod {
  GPSM_BIN_SEARCH,
};

//////// =========== Multi-way intersect ===============

/**
1. InstanceGatherFunctor are used in the following interfaces.
instance_gather_functor(path_id, M), where M is the partial instance.
instance_gather_functor is defined by the user to build the partial instance,
i.e., M.
2. The connectivity and ordering constraints, as well as the check for duplicate
removal, may be passed as arguments to facilitate the intersection.
3. The results are written to output_row_ptrs and outptu_cols, which may be the
type T* or DeviceArray<T>*.
**/

/////////// The most general interface for multi-way intersect. //////////
// Currently, only GPSM_BIN_SEARCH for this interface is supported.
template <ProcessMethod method = GPSM_BIN_SEARCH, bool kCheckCondition = true,
          bool kCheckDuplicates = true, typename InstanceGatherFunctor,
          typename IndexType, typename uintE, typename uintV>
void Intersect(InstanceGatherFunctor instance_gather_functor, size_t path_num,
               uintE *row_ptrs, uintV *cols, DevConnType *conn,
               DevCondArrayType *cond, uintP *d_partition_ids,
               const uintV prime_edge_v0, const uintV prime_edge_v1,
               IndexType *&output_row_ptrs, uintV *&output_cols,
               CudaContext *context) {
  if (method == GPSM_BIN_SEARCH) {
    GpsmBinSearch<kCheckCondition, kCheckDuplicates>(
        instance_gather_functor, path_num, row_ptrs, cols, conn, cond,
        d_partition_ids, prime_edge_v0, prime_edge_v1, output_row_ptrs,
        output_cols, context);
  } else {
    assert(false);
  }
}
// with DeviceArray as the input
template <ProcessMethod method = GPSM_BIN_SEARCH, bool kCheckCondition = true,
          bool kCheckDuplicates = true, typename InstanceGatherFunctor,
          typename IndexType, typename uintE, typename uintV>
void Intersect(InstanceGatherFunctor instance_gather_functor, size_t path_num,
               uintE *row_ptrs, uintV *cols, DevConnType *conn,
               DevCondArrayType *cond, uintP *d_partition_ids,
               const uintV prime_edge_v0, const uintV prime_edge_v1,
               DeviceArray<IndexType> *&output_row_ptrs,
               DeviceArray<uintV> *&output_cols, CudaContext *context) {
  IndexType *output_row_ptrs_data = NULL;
  uintV *output_cols_data = NULL;

  Intersect<method, kCheckCondition, kCheckDuplicates>(
      instance_gather_functor, path_num, row_ptrs, cols, conn, cond,
      d_partition_ids, prime_edge_v0, prime_edge_v1, output_row_ptrs_data,
      output_cols_data, context);

  output_row_ptrs = new DeviceArray<IndexType>(output_row_ptrs_data,
                                               path_num + 1, context, true);
  IndexType total_output_cols_size;
  DToH(&total_output_cols_size, output_row_ptrs_data + path_num, 1);
  output_cols = new DeviceArray<uintV>(output_cols_data, total_output_cols_size,
                                       context, true);
}

/////////// No duplicate removal check. ///////////
template <ProcessMethod method = GPSM_BIN_SEARCH, bool kCheckCondition = true,
          typename InstanceGatherFunctor, typename IndexType, typename uintE,
          typename uintV>
void Intersect(InstanceGatherFunctor instance_gather_functor, size_t path_num,
               uintE *row_ptrs, uintV *cols, DevConnType *conn,
               DevCondArrayType *cond, IndexType *&output_row_ptrs,
               uintV *&output_cols, CudaContext *context) {
  if (method == GPSM_BIN_SEARCH) {
    GpsmBinSearch<kCheckCondition, false>(
        instance_gather_functor, path_num, row_ptrs, cols, conn, cond,
        (uintP *)NULL, (uintV)0, (uintV)0, output_row_ptrs, output_cols,
        context);
  } else {
    assert(false);
  }
}

// with DeviceArray as the input
template <ProcessMethod method = GPSM_BIN_SEARCH, bool kCheckCondition = true,
          typename InstanceGatherFunctor, typename IndexType, typename uintE,
          typename uintV>
void Intersect(InstanceGatherFunctor instance_gather_functor, size_t path_num,
               uintE *row_ptrs, uintV *cols, DevConnType *conn,
               DevCondArrayType *cond, DeviceArray<IndexType> *&output_row_ptrs,
               DeviceArray<uintV> *&output_cols, CudaContext *context) {
  IndexType *output_row_ptrs_data = NULL;
  uintV *output_cols_data = NULL;

  Intersect<method, kCheckCondition>(
      instance_gather_functor, path_num, row_ptrs, cols, conn, cond,
      output_row_ptrs_data, output_cols_data, context);

  output_row_ptrs = new DeviceArray<IndexType>(output_row_ptrs_data,
                                               path_num + 1, context, true);
  IndexType total_output_cols_size;
  DToH(&total_output_cols_size, output_row_ptrs_data + path_num, 1);
  output_cols = new DeviceArray<uintV>(output_cols_data, total_output_cols_size,
                                       context, true);
}

//////// =========== Multi-way intersect count ===============

/**
The difference from multi-way intersect is that the intersect count APIs do not
materialize the output reuslts but only output the aggregated counts for each
path or the overall counts.
1. InstanceGatherFunctor are used in the following interfaces.
instance_gather_functor(path_id, M), where M is the partial instance.
instance_gather_functor is defined by the user to build the partial instance,
i.e., M.
2. The connectivity and ordering constraints, as well as the check for duplicate
removal, may be passed as arguments to facilitate the intersection.
3. CountFunctor returns the added count for a given path.
count_functor (path_id, candidate) -> size_t (count)
**/

// Return the total aggregated count.
template <ProcessMethod method = GPSM_BIN_SEARCH, bool kCheckCondition = true,
          bool kCheckDuplicates = true, typename InstanceGatherFunctor,
          typename CountFunctor, typename uintE, typename uintV>
size_t IntersectCount(InstanceGatherFunctor instance_gather_functor,
                      size_t path_num, uintE *row_ptrs, uintV *cols,
                      DevConnType *conn, DevCondArrayType *cond,
                      uintP *d_partition_ids, const uintV prime_edge_v0,
                      const uintV prime_edge_v1, CountFunctor count_functor,
                      CudaContext *context) {
  size_t ret = 0;
  size_t *dummy_output_offsets = NULL;
  if (method == GPSM_BIN_SEARCH) {
    ret = GpsmBinSearchCount<true, kCheckCondition, kCheckDuplicates>(
        instance_gather_functor, path_num, row_ptrs, cols, conn, cond,
        d_partition_ids, prime_edge_v0, prime_edge_v1, count_functor,
        dummy_output_offsets, context);
  } else {
    assert(false);
  }
  return ret;
}

// Multi-way intersect and count the intersection result for each path,
// After counting the result for each path, perform prefix scan and write
// the result to output_offsets.
template <ProcessMethod method = GPSM_BIN_SEARCH, bool kCheckCondition = true,
          bool kCheckDuplicates = true, typename InstanceGatherFunctor,
          typename CountFunctor, typename IndexType, typename uintE,
          typename uintV>
void IntersectCount(InstanceGatherFunctor instance_gather_functor,
                    size_t path_num, uintE *row_ptrs, uintV *cols,
                    DevConnType *conn, DevCondArrayType *cond,
                    uintP *d_partition_ids, const uintV prime_edge_v0,
                    const uintV prime_edge_v1, CountFunctor count_functor,
                    DeviceArray<IndexType> *&output_offsets,
                    CudaContext *context) {
  IndexType *output_offsets_data = NULL;
  if (method == GPSM_BIN_SEARCH) {
    GpsmBinSearchCount<false, kCheckCondition, kCheckDuplicates>(
        instance_gather_functor, path_num, row_ptrs, cols, conn, cond,
        d_partition_ids, prime_edge_v0, prime_edge_v1, count_functor,
        output_offsets_data, context);
  } else {
    assert(false);
  }
  output_offsets = new DeviceArray<IndexType>(output_offsets_data, path_num + 1,
                                              context, true);
}

}  // namespace Intersect
}  // namespace GpuUtils
#endif
