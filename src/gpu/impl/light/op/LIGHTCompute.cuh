#ifndef __HYBRID_LIGHT_PIPELINE_GPU_COMPONENT_COMPUTE_CUH__
#define __HYBRID_LIGHT_PIPELINE_GPU_COMPONENT_COMPUTE_CUH__

#include "LIGHTCommon.cuh"

#include "DevGraphPartition.cuh"
#include "Intersect.cuh"
#include "LazyTraversalPlan.h"

namespace Light {
template <bool kInterPartSearch>
static void ComputeGeneral(LightWorkContext* wctx, uintV cur_level,
                           size_t path_num) {
  auto d_partition_id = wctx->d_partition_id;
  auto context = wctx->context;
  auto im_data = wctx->im_data;
  auto im_data_holder = wctx->im_data_holder;
  auto dev_plan = wctx->dev_plan;
  auto graph_partition = wctx->graph_partition;
  auto gpu_profiler = wctx->gpu_profiler;

  auto& d_instances = im_data->GetInstances();
  auto& d_candidates = im_data->GetCandidates();
  auto& d_candidates_offsets = im_data->GetCandidatesOffsets();
  auto& d_candidates_indices = im_data->GetCandidatesIndices();
  uintP* d_partition_ids = NULL;
  if (kInterPartSearch) {
    auto graph_dev_tracker = wctx->graph_dev_tracker;
    d_partition_ids = graph_dev_tracker->GetPartitionIds()->GetArray();
  }

  im_data_holder->GatherImData(im_data, context);
  auto d_seq_instances = im_data_holder->GetSeqInstances()->GetArray();

  uintE* row_ptrs = graph_partition->GetRowPtrs()->GetArray();
  uintV* cols = graph_partition->GetCols()->GetArray();

  auto backward_conn = dev_plan->GetBackwardConnectivity()->GetArray();
  auto computed_order = dev_plan->GetComputedOrdering()->GetArray();

  // instance_gather_functor
  auto instance_gather_functor = [=] DEVICE(int index, uintV* M) {
    if (kInterPartSearch) {
      // have to load the first two edges to check duplicates
      // The first two pattern vertices are guaranteed to be 0, 1
      M[0] = d_seq_instances[0][index];
      M[1] = d_seq_instances[1][index];
    }
    auto& cond = computed_order[cur_level];
    for (size_t i = 0; i < cond.GetCount(); ++i) {
      auto u = cond.Get(i).GetOperand();
      M[u] = d_seq_instances[u][index];
    }
  };

  gpu_profiler->StartTimer("compute_set_intersect_time", d_partition_id,
                           context->Stream());
  GpuUtils::Intersect::Intersect<
      GpuUtils::Intersect::ProcessMethod::GPSM_BIN_SEARCH, true,
      kInterPartSearch>(instance_gather_functor, path_num, row_ptrs, cols,
                        backward_conn + cur_level, computed_order + cur_level,
                        d_partition_ids, (uintV)0, (uintV)1,
                        d_candidates_offsets[cur_level],
                        d_candidates[cur_level], context);
  gpu_profiler->EndTimer("compute_set_intersect_time", d_partition_id,
                         context->Stream());

  ReAllocate(d_candidates_indices[cur_level], path_num, context);
  GpuUtils::Transform::Sequence(d_candidates_indices[cur_level]->GetArray(),
                                path_num, (size_t)0, context);
}

}  // namespace Light

#endif
