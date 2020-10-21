#ifndef __HYBRID_LIGHT_GPU_PIPELINE_COMPONENT_COUNT_CUH__
#define __HYBRID_LIGHT_GPU_PIPELINE_COMPONENT_COUNT_CUH__

#include "VerifyLIGHTCount.cuh"

#include "DevLazyTraversalPlan.cuh"
#include "GPUUtil.cuh"

#include "GPUBinSearch.cuh"
#include "Intersect.cuh"

#include "LazyTraversalPlan.h"
#include "Meta.h"

namespace Light {

// For two pattern vertices that have been computed,
// i.e., with their candidate sets available,
// count the total count, given any any condition constraints
// between them.
template <bool from_index>
static void GeneralCountTwoComputedVertices(
    size_t d_partition_id, uintV u0, uintV u1, const CondOperator cond_operator,
    size_t path_num, size_t* candidates_indices0_data,
    size_t* candidates_indices1_data, size_t* candidates_offsets0_data,
    size_t* candidates_offsets1_data, uintV* candidates0_data,
    uintV* candidates1_data, size_t* d_total_count_data, CudaContext* context,
    GPUProfiler* gpu_profiler, CountProfiler* count_profiler) {
  assert(candidates0_data && candidates1_data);
  DeviceArray<size_t> workload_offsets(path_num + 1, context);
  size_t* workload_offsets_data = workload_offsets.GetArray();

  GpuUtils::Scan::TransformScan(
      [=] DEVICE(int index) {
        size_t p0 = from_index ? candidates_indices0_data[index] : index;
        size_t count0 =
            candidates_offsets0_data[p0 + 1] - candidates_offsets0_data[p0];
        size_t p1 = from_index ? candidates_indices1_data[index] : index;
        size_t count1 =
            candidates_offsets1_data[p1 + 1] - candidates_offsets1_data[p1];
        size_t ret;
        if (count0 <= count1) {
          ret = count0;
        } else {
          ret = count1;
        }
        return ret;
      },
      path_num, workload_offsets.GetArray(),
      workload_offsets.GetArray() + path_num, context);

  size_t total_workload_count;
  DToH(&total_workload_count, workload_offsets.GetArray() + path_num, 1);
  assert((int)total_workload_count == total_workload_count);

  DeviceArray<size_t> path_ids(total_workload_count, context);
  size_t* path_ids_data = path_ids.GetArray();

  GpuUtils::LoadBalance::LoadBalanceSearch<MGPULaunchBoxVT1>(
      total_workload_count, workload_offsets.GetArray(), path_num,
      path_ids.GetArray(), context);

  auto k = [=] DEVICE(int index) {
    size_t path_id = path_ids_data[index];
    size_t rank = index - workload_offsets_data[path_id];

    size_t p0 = from_index ? candidates_indices0_data[path_id] : path_id;
    size_t count0 =
        candidates_offsets0_data[p0 + 1] - candidates_offsets0_data[p0];
    size_t p1 = from_index ? candidates_indices1_data[path_id] : path_id;
    size_t count1 =
        candidates_offsets1_data[p1 + 1] - candidates_offsets1_data[p1];

    uintV* search_array;
    size_t search_count;
    uintV search_element;
    if (count0 <= count1) {
      search_array = candidates1_data + candidates_offsets1_data[p1];
      search_count = count1;
      search_element = candidates0_data[candidates_offsets0_data[p0] + rank];
    } else {
      search_array = candidates0_data + candidates_offsets0_data[p0];
      search_count = count0;
      search_element = candidates1_data[candidates_offsets1_data[p1] + rank];
    }
    size_t bin_pos = GpuUtils::BinSearch::BinSearch(search_array, search_count,
                                                    search_element);
    bool equal =
        (bin_pos < search_count && search_array[bin_pos] == search_element);

    size_t ret = 0;
    if (cond_operator == LESS_THAN) {
      if (count0 <= count1) {
        bin_pos += equal ? 1 : 0;
        ret = search_count - bin_pos;
      } else {
        ret = bin_pos;
      }
    } else if (cond_operator == LARGER_THAN) {
      if (count0 <= count1) {
        ret = bin_pos;
      } else {
        bin_pos += equal ? 1 : 0;
        ret = search_count - bin_pos;
      }
    } else {
      // NON_EQUAL
      ret = search_count;
      ret -= equal ? 1 : 0;
    }
    return ret;
  };

  GpuUtils::Reduce::TransformReduce(k, total_workload_count, d_total_count_data,
                                    context);
}

static size_t LIGHTCount(LightWorkContext* wctx, size_t cur_exec_level) {
  auto d_partition_id = wctx->d_partition_id;
  auto context = wctx->context;
  auto plan = wctx->plan;
  auto im_data = wctx->im_data;
  auto im_data_holder = wctx->im_data_holder;
  auto dev_plan = wctx->dev_plan;
  auto gpu_profiler = wctx->gpu_profiler;
  auto count_profiler = wctx->count_profiler;

  // intermediate data
  auto& d_instances = im_data->GetInstances();
  auto& d_candidates = im_data->GetCandidates();
  auto& d_candidates_offsets = im_data->GetCandidatesOffsets();
  auto& d_candidates_indices = im_data->GetCandidatesIndices();

  im_data_holder->GatherImData(im_data, context);
  auto d_seq_instances = im_data_holder->GetSeqInstances()->GetArray();
  auto d_seq_candidates = im_data_holder->GetSeqCandidates()->GetArray();
  auto d_seq_candidates_indices =
      im_data_holder->GetSeqCandidatesIndices()->GetArray();
  auto d_seq_candidates_offsets =
      im_data_holder->GetSeqCandidatesOffsets()->GetArray();

  // materialized_vertices and unmaterialized_vertices
  auto& materialized_vertices = plan->GetMaterializedVertices()[cur_exec_level];
  auto& unmaterialized_vertices =
      plan->GetComputedUnmaterializedVertices()[cur_exec_level];

  auto d_materialized_vertices =
      dev_plan->GetMaterializedVertices()->GetArray() + cur_exec_level;
  auto d_unmaterialized_vertices =
      dev_plan->GetComputedUnmaterializedVertices()->GetArray() +
      cur_exec_level;

  size_t path_num = d_instances[materialized_vertices[0]]->GetSize();
  DeviceArray<size_t> total_count(1, context);

  if (unmaterialized_vertices.size() == 1) {
    uintV cur_level = unmaterialized_vertices[0];
    size_t* cur_candidates_indices =
        d_candidates_indices[cur_level]->GetArray();
    size_t* cur_candidates_offsets =
        d_candidates_offsets[cur_level]->GetArray();
    auto k = [=] DEVICE(int index) {
      size_t p = cur_candidates_indices[index];
      size_t count = cur_candidates_offsets[p + 1] - cur_candidates_offsets[p];
      return count;
    };
    GpuUtils::Reduce::TransformReduce(k, path_num, total_count.GetArray(),
                                      context);

  } else if (unmaterialized_vertices.size() == 2) {
    uintV u0 = unmaterialized_vertices[0];
    uintV u1 = unmaterialized_vertices[1];
    CondOperator cond_operator =
        Plan::GetConditionType(u0, u1, plan->GetOrdering());
    CUDA_ERROR(cudaMemsetAsync(total_count.GetArray(), 0, sizeof(size_t),
                               context->Stream()));
    if (d_candidates[u0]->GetSize() > 0 && d_candidates[u1]->GetSize() > 0) {
      GeneralCountTwoComputedVertices<true>(
          d_partition_id, u0, u1, cond_operator, path_num,
          d_candidates_indices[u0]->GetArray(),
          d_candidates_indices[u1]->GetArray(),
          d_candidates_offsets[u0]->GetArray(),
          d_candidates_offsets[u1]->GetArray(), d_candidates[u0]->GetArray(),
          d_candidates[u1]->GetArray(), total_count.GetArray(), context,
          gpu_profiler, count_profiler);
    }

  } else {
    assert(false);
  }

  size_t ret;
  DToH(&ret, total_count.GetArray(), 1);
  return ret;
}

}  // namespace Light

#endif
