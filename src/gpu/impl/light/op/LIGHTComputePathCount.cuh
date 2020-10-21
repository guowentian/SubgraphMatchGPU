#ifndef __HYBRID_LIGHT_PIPELINE_GPU_COMPONENT_COMPUTE_PATH_COUNT_CUH__
#define __HYBRID_LIGHT_PIPELINE_GPU_COMPONENT_COMPUTE_PATH_COUNT_CUH__

#include "LIGHTCommon.cuh"

#include "DevGraphPartition.cuh"
#include "Intersect.cuh"
#include "LazyTraversalPlan.h"

namespace Light {
static void ComputePathCount(LightWorkContext* wctx, size_t cur_exec_level) {
  auto d_partition_id = wctx->d_partition_id;
  auto context = wctx->context;
  auto plan = wctx->plan;
  auto im_data = wctx->im_data;
  auto im_data_holder = wctx->im_data_holder;
  auto graph_partition = wctx->graph_partition;
  auto dev_plan = wctx->dev_plan;
  auto gpu_profiler = wctx->gpu_profiler;

  auto& d_instances = im_data->GetInstances();
  auto& d_candidates = im_data->GetCandidates();
  auto& d_candidates_offsets = im_data->GetCandidatesOffsets();
  auto& d_candidates_indices = im_data->GetCandidatesIndices();

  im_data_holder->GatherImData(im_data, context);
  auto d_seq_instances = im_data_holder->GetSeqInstances()->GetArray();

  uintV materialized_vertex =
      plan->GetMaterializedVertices()[cur_exec_level][0];
  size_t path_num = d_instances[materialized_vertex]->GetSize();
  uintE* row_ptrs = graph_partition->GetRowPtrs()->GetArray();
  uintV* cols = graph_partition->GetCols()->GetArray();
  size_t graph_vertex_count = graph_partition->GetVertexCount();

  auto backward_conn = dev_plan->GetBackwardConnectivity()->GetArray();
  auto computed_order = dev_plan->GetComputedOrdering()->GetArray();

  auto& exec_seq = plan->GetExecuteOperations();
  uintV cur_level = exec_seq[cur_exec_level].second;

  // instance_gather_functor
  auto instance_gather_functor = [=] DEVICE(int index, uintV* M) {
    auto& cond = computed_order[cur_level];
    for (size_t i = 0; i < cond.GetCount(); ++i) {
      auto u = cond.Get(i).GetOperand();
      M[u] = d_seq_instances[u][index];
    }
  };
  auto count_functor = [=] DEVICE(size_t path_id, uintV candidate) {
    return 1;
  };

  gpu_profiler->StartTimer("compute_set_intersect_time", d_partition_id,
                           context->Stream());
  GpuUtils::Intersect::IntersectCount<
      GpuUtils::Intersect::ProcessMethod::GPSM_BIN_SEARCH, true, false>(
      instance_gather_functor, path_num, row_ptrs, cols,
      backward_conn + cur_level, computed_order + cur_level, (uintP*)NULL,
      (uintV)0, (uintV)0, count_functor, d_candidates_offsets[cur_level],
      context);
  gpu_profiler->EndTimer("compute_set_intersect_time", d_partition_id,
                         context->Stream());

  ReAllocate(d_candidates_indices[cur_level], path_num, context);
  GpuUtils::Transform::Sequence(d_candidates_indices[cur_level]->GetArray(),
                                path_num, (size_t)0, context);
}

}  // namespace Light
#endif