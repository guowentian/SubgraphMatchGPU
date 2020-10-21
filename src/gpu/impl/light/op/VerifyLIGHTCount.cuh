#ifndef __HYBRID_LIGHT_GPU_PIPELINE_COMPONENT_VERIFY_COUNT_CUH__
#define __HYBRID_LIGHT_GPU_PIPELINE_COMPONENT_VERIFY_COUNT_CUH__

#include "GPUUtil.cuh"

#include "GPUBinSearch.cuh"
#include "Intersect.cuh"

namespace Light {
template <bool from_index>
static size_t VerifyGeneralCountTwoComputedVertices(
    size_t d_partition_id, uintV u0, uintV u1, const CondOperator cond_operator,
    size_t path_num, size_t* candidates_indices0_data,
    size_t* candidates_indices1_data, size_t* candidates_offsets0_data,
    size_t* candidates_offsets1_data, uintV* candidates0_data,
    uintV* candidates1_data, size_t* d_total_count_data,
    CudaContext* context, GPUProfiler* gpu_profiler,
    CountProfiler* count_profiler) {
  size_t* h_candidates_indices0 = NULL;
  size_t* h_candidates_indices1 = NULL;
  if (from_index) {
    h_candidates_indices0 = new size_t[path_num];
    h_candidates_indices1 = new size_t[path_num];
    DToH(h_candidates_indices0, candidates_indices0_data, path_num);
    DToH(h_candidates_indices1, candidates_indices1_data, path_num);
  }

  size_t* h_candidates_offsets0 = new size_t[path_num + 1];
  size_t* h_candidates_offsets1 = new size_t[path_num + 1];
  DToH(h_candidates_offsets0, candidates_offsets0_data, path_num + 1);
  DToH(h_candidates_offsets1, candidates_offsets1_data, path_num + 1);

  size_t total_candidates_count0 = h_candidates_offsets0[path_num];
  size_t total_candidates_count1 = h_candidates_offsets1[path_num];
  uintV* h_candidates0 = new uintV[total_candidates_count0];
  uintV* h_candidates1 = new uintV[total_candidates_count1];
  DToH(h_candidates0, candidates0_data, total_candidates_count0);
  DToH(h_candidates1, candidates1_data, total_candidates_count1);

  size_t ret = 0;
  for (size_t p = 0; p < path_num; ++p) {
    size_t p0 = from_index ? h_candidates_indices0[p] : p;
    size_t p1 = from_index ? h_candidates_indices1[p] : p;
    for (size_t i0 = h_candidates_offsets0[p0];
         i0 < h_candidates_offsets0[p0 + 1]; ++i0) {
      size_t v0 = h_candidates0[i0];
      for (size_t i1 = h_candidates_offsets1[p1];
           i1 < h_candidates_offsets1[p1 + 1]; ++i1) {
        uintV v1 = h_candidates1[i1];
        if (cond_operator == LESS_THAN) {
          if (v0 < v1) ret++;
        } else if (cond_operator == LARGER_THAN) {
          if (v0 > v1) ret++;
        } else {
          if (v0 != v1) ret++;
        }
      }
    }
  }

  delete[] h_candidates0;
  h_candidates0 = NULL;
  delete[] h_candidates1;
  h_candidates1 = NULL;

  delete[] h_candidates_offsets0;
  h_candidates_offsets0 = NULL;
  delete[] h_candidates_offsets1;
  h_candidates_offsets1 = NULL;

  if (from_index) {
    delete[] h_candidates_indices0;
    h_candidates_indices0 = NULL;
    delete[] h_candidates_indices1;
    h_candidates_indices1 = NULL;
  }

  return ret;
}
}  // namespace Light

#endif