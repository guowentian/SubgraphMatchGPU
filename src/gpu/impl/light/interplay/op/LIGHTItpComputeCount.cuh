#ifndef __HYBRID_LIGHT_PIPELINE_GPU_COMPONENT_ITP_COMPUTE_COUNT_CUH__
#define __HYBRID_LIGHT_PIPELINE_GPU_COMPONENT_ITP_COMPUTE_COUNT_CUH__

#include "Iterator.cuh"
#include "LIGHTComputeCount.cuh"
#include "LIGHTFilterCompute.cuh"
#include "LIGHTItpComputeCount.cuh"
#include "LIGHTItpCount.cuh"

namespace Light {

static size_t ItpComputeCount(LightItpWorkContext* wctx,
                              size_t cur_exec_level) {
  auto dfs_id = wctx->dfs_id;
  auto d_partition_id = wctx->d_partition_id;
  auto context = wctx->context;
  auto plan = wctx->plan;
  auto im_data = wctx->im_data;
  auto im_data_holder = wctx->im_data_holder;
  auto graph_partition = wctx->graph_partition;
  auto dev_plan = wctx->dev_plan;
  auto graph_dev_tracker = wctx->graph_dev_tracker;
  auto gpu_profiler = wctx->gpu_profiler;

  // partial instances
  auto& d_instances = im_data->GetInstances();
  auto& d_candidates = im_data->GetCandidates();
  auto& d_candidates_offsets = im_data->GetCandidatesOffsets();
  auto& d_candidates_indices = im_data->GetCandidatesIndices();
  uintP* d_partition_ids = graph_dev_tracker->GetPartitionIds()->GetArray();
  im_data_holder->GatherImData(im_data, context);
  auto d_seq_instances = im_data_holder->GetSeqInstances()->GetArray();

  // graph
  uintE* row_ptrs = graph_partition->GetRowPtrs()->GetArray();
  uintV* cols = graph_partition->GetCols()->GetArray();
  size_t graph_vertex_count = graph_partition->GetVertexCount();

  // (inter-partition) plan related data
  auto& materialized_vertices =
      plan->GetInterPartitionMaterializedVertices()[dfs_id][cur_exec_level];
  auto& unmaterialized_vertices =
      plan->GetInterPartitionComputedUnmaterializedVertices()[dfs_id]
                                                             [cur_exec_level];

  auto& dfs_orders = plan->GetSearchSequences();
  AllCondType conditions;
  Plan::GetIndexBasedOrdering(conditions, plan->GetOrdering(),
                              dfs_orders[dfs_id]);

  auto backward_conn = dev_plan->GetBackwardConnectivity()->GetArray();
  auto computed_order = dev_plan->GetComputedOrdering()->GetArray();

  size_t path_num = d_instances[materialized_vertices[0]]->GetSize();
  uintV cur_level =
      plan->GetInterPartitionExecuteOperations()[dfs_id][cur_exec_level].second;

  // for the partial instances formed by the materialized vertices, enforce the
  // ordering
  DeviceArray<size_t>* valid_path_ids = NULL;
  ItpEnforceOrderingPartialInstances(wctx, cur_exec_level, path_num,
                                     valid_path_ids);
  if (!valid_path_ids->GetSize()) {
    delete valid_path_ids;
    valid_path_ids = NULL;
    return 0;
  }

  size_t* valid_path_ids_array = valid_path_ids->GetArray();
  // instance_gather_functor
  auto instance_gather_functor = [=] DEVICE(int index, uintV* M) {
    size_t path_id = valid_path_ids_array[index];
    // have to load the first two edges to check duplicates
    // The first two pattern vertices are guaranteed to be 0, 1
    M[0] = d_seq_instances[0][path_id];
    M[1] = d_seq_instances[1][path_id];

    auto& cond = computed_order[cur_level];
    for (size_t i = 0; i < cond.GetCount(); ++i) {
      auto u = cond.Get(i).GetOperand();
      M[u] = d_seq_instances[u][path_id];
    }
  };

  gpu_profiler->StartTimer("compute_set_intersect_time", d_partition_id,
                           context->Stream());
  gpu_profiler->StartTimer("compute_count_set_intersect_count_time",
                           d_partition_id, context->Stream());

  size_t ret = 0;
  if (unmaterialized_vertices.size() == 0) {
    auto count_functor = [=] DEVICE(size_t path_id, uintV candidate) {
      return 1;
    };

    ret = GpuUtils::Intersect ::IntersectCount<
        GpuUtils::Intersect::GPSM_BIN_SEARCH, true, true>(
        instance_gather_functor, valid_path_ids->GetSize(), row_ptrs, cols,
        backward_conn + cur_level, computed_order + cur_level, d_partition_ids,
        (uintV)0, (uintV)1, count_functor, context);

  } else if (unmaterialized_vertices.size() == 1) {
    // enforce ordering between other_u and the materialized vertices
    uintV other_u = unmaterialized_vertices[0];
    auto other_cond =
        dev_plan->GetCountToMaterializedOrdering()->GetArray() + other_u;
    size_t* valid_path_ids_array = valid_path_ids->GetArray();
    auto path_iterator = [=] DEVICE(int index) {
      return valid_path_ids_array[index];
    };
    DeviceArray<size_t>* other_filter_candidates_offsets = NULL;
    DeviceArray<uintV>* other_filter_candidates = NULL;

    FilterCandidate(
        wctx, other_u,
        GpuUtils::Iterator::MakeLoadIterator<size_t>(path_iterator, 0),
        valid_path_ids->GetSize(), other_cond, other_filter_candidates_offsets,
        other_filter_candidates);

    if (other_filter_candidates->GetSize() > 0) {
      // enforce ordering between other_u and cur_level
      uintV* other_candidates = other_filter_candidates->GetArray();
      size_t* other_candidates_offsets =
          other_filter_candidates_offsets->GetArray();
      CondOperator cond_operator =
          Plan::GetConditionType(cur_level, other_u, conditions);

      auto count_functor = [=] DEVICE(size_t path_id, uintV candidate) {
        uintV* search_array =
            other_candidates + other_candidates_offsets[path_id];
        size_t search_count = other_candidates_offsets[path_id + 1] -
                              other_candidates_offsets[path_id];

        return ThreadCountWithComputedVertex(search_array, search_count,
                                             candidate, cond_operator);
      };
      ret = GpuUtils::Intersect ::IntersectCount<
          GpuUtils::Intersect::GPSM_BIN_SEARCH, true, true>(
          instance_gather_functor, valid_path_ids->GetSize(), row_ptrs, cols,
          backward_conn + cur_level, computed_order + cur_level,
          d_partition_ids, (uintV)0, (uintV)1, count_functor, context);
    }

    delete other_filter_candidates;
    other_filter_candidates = NULL;
    delete other_filter_candidates_offsets;
    other_filter_candidates_offsets = NULL;

  } else {
    assert(false);
  }

  gpu_profiler->EndTimer("compute_count_set_intersect_count_time",
                         d_partition_id, context->Stream());

  gpu_profiler->EndTimer("compute_set_intersect_time", d_partition_id,
                         context->Stream());

  delete valid_path_ids;
  valid_path_ids = NULL;

  return ret;
}

}  // namespace Light

#endif