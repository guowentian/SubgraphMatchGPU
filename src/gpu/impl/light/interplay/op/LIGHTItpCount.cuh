#ifndef __HYBRID_LIGHT_INTERPLAY_PIPELINE_GPU_COMPONENT_COUNT_CUH__
#define __HYBRID_LIGHT_INTERPLAY_PIPELINE_GPU_COMPONENT_COUNT_CUH__

#include "LIGHTCount.cuh"
#include "VerifyLIGHTItpCount.cuh"

#include "GraphDevTracker.cuh"
#include "Iterator.cuh"
#include "LIGHTFilterCompute.cuh"
#include "LIGHTItpCommon.cuh"

namespace Light {
static void ItpEnforceOrderingPartialInstances(
    LightWorkContext* wctx, size_t cur_exec_level, size_t path_num,
    DeviceArray<size_t>*& valid_path_ids) {
  auto context = wctx->context;
  auto im_data_holder = wctx->im_data_holder;
  auto dev_plan = wctx->dev_plan;

  auto d_seq_instances = im_data_holder->GetSeqInstances()->GetArray();
  auto d_materialized_vertices =
      dev_plan->GetMaterializedVertices()->GetArray() + cur_exec_level;
  auto d_count_to_materialized_order =
      dev_plan->GetCountToMaterializedOrdering()->GetArray();

  auto select_functor = [=] DEVICE(size_t index) {
    uintV M[kMaxQueryVerticesNum] = {kMaxuintV};
    for (size_t i = 0; i < d_materialized_vertices->GetCount(); ++i) {
      uintV u = d_materialized_vertices->Get(i);
      M[u] = d_seq_instances[u][index];
    }

    bool ret = true;
    for (size_t i = 0; i < d_materialized_vertices->GetCount(); ++i) {
      uintV u = d_materialized_vertices->Get(i);
      auto& cond = d_count_to_materialized_order[u];
      if (!ThreadCheckCondition(cond, M, M[u])) {
        ret = false;
        break;
      }
    }
    return ret;
  };

  int compact_path_num = 0;
  GpuUtils::Compact::Compact(
      GpuUtils::Iterator::CountingIterator<size_t>(0), path_num,
      GpuUtils::Iterator::MakeLoadIterator<bool>(select_functor, (size_t)0),
      valid_path_ids, compact_path_num, context);
}

static size_t ItpGeneralCountOneVertex(LightItpWorkContext* wctx,
                                       size_t cur_exec_level,
                                       DeviceArray<size_t>* valid_path_ids) {
  auto dfs_id = wctx->dfs_id;
  auto context = wctx->context;
  auto plan = wctx->plan;
  auto im_data = wctx->im_data;
  auto im_data_holder = wctx->im_data_holder;
  auto dev_plan = wctx->dev_plan;

  assert(valid_path_ids->GetSize());
  auto& unmaterialized_vertices =
      plan->GetInterPartitionComputedUnmaterializedVertices()[dfs_id]
                                                             [cur_exec_level];
  uintV cur_level = unmaterialized_vertices[0];

  auto cur_candidates_indices =
      im_data->GetCandidatesIndices()[cur_level]->GetArray();
  auto cur_candidates_offsets =
      im_data->GetCandidatesOffsets()[cur_level]->GetArray();
  auto cur_candidates = im_data->GetCandidates()[cur_level]->GetArray();
  auto valid_path_ids_array = valid_path_ids->GetArray();

  auto d_materialized_vertices =
      dev_plan->GetMaterializedVertices()->GetArray() + cur_exec_level;
  auto d_seq_instances = im_data_holder->GetSeqInstances()->GetArray();
  auto cond =
      dev_plan->GetCountToMaterializedOrdering()->GetArray() + cur_level;

  // warp based aggregate
  auto select_functor = [=] DEVICE(size_t thread_index) {
    size_t warp_index = thread_index / THREADS_PER_WARP;
    size_t valid_path_id = valid_path_ids_array[warp_index];
    size_t path_id = cur_candidates_indices[valid_path_id];

    uintV M[kMaxQueryVerticesNum] = {kMaxuintV};
    for (size_t i = 0; i < d_materialized_vertices->GetCount(); ++i) {
      uintV u = d_materialized_vertices->Get(i);
      M[u] = d_seq_instances[u][path_id];
    }

    size_t ret = 0;
    size_t off = thread_index % THREADS_PER_WARP;
    for (size_t cand_index = cur_candidates_offsets[path_id] + off;
         cand_index < cur_candidates_offsets[path_id + 1];
         cand_index += THREADS_PER_WARP) {
      uintV candidate = cur_candidates[cand_index];
      bool valid = ThreadCheckCondition(*cond, M, candidate);
      if (valid) ++ret;
    }
    return ret;
  };

  DeviceArray<size_t> d_total_count(1, context);
  GpuUtils::Reduce::TransformReduce(
      select_functor, valid_path_ids->GetSize() * THREADS_PER_WARP,
      d_total_count.GetArray(), context);
  size_t ret = 0;
  DToH(&ret, d_total_count.GetArray(), 1);
  return ret;
}

static size_t ItpGeneralCountTwoVertices(LightItpWorkContext* wctx,
                                         size_t cur_exec_level,
                                         DeviceArray<size_t>* valid_path_ids) {
  auto dfs_id = wctx->dfs_id;
  auto d_partition_id = wctx->d_partition_id;
  auto context = wctx->context;
  auto plan = wctx->plan;
  auto dev_plan = wctx->dev_plan;
  auto gpu_profiler = wctx->gpu_profiler;
  auto count_profiler = wctx->count_profiler;

  assert(valid_path_ids->GetSize());
  auto& unmaterialized_vertices =
      plan->GetInterPartitionComputedUnmaterializedVertices()[dfs_id]
                                                             [cur_exec_level];

  auto& dfs_orders = plan->GetSearchSequences();
  AllCondType index_based_cond;
  Plan::GetIndexBasedOrdering(index_based_cond, plan->GetOrdering(),
                              dfs_orders[dfs_id]);

  // apply ordering constraints for each vertex separately
  // and materialize the new candidate set
  std::vector<DeviceArray<size_t>*> filter_candidates_offsets(2, NULL);
  std::vector<DeviceArray<uintV>*> filter_candidates(2, NULL);
  for (size_t i = 0; i < unmaterialized_vertices.size(); ++i) {
    uintV u = unmaterialized_vertices[i];
    DevCondArrayType* d_cond =
        dev_plan->GetCountToMaterializedOrdering()->GetArray() + u;
    size_t* valid_path_ids_array = valid_path_ids->GetArray();
    auto path_iterator = [=] DEVICE(size_t index) {
      return valid_path_ids_array[index];
    };
    FilterCandidate(
        wctx, u, GpuUtils::Iterator::MakeLoadIterator<size_t>(path_iterator, 0),
        valid_path_ids->GetSize(), d_cond, filter_candidates_offsets[i],
        filter_candidates[i]);
#if defined(DEBUG)
    /*VerifyFiliterCandidate(
        d_partition_id, context, u, valid_path_ids, valid_path_ids->GetSize(),
        filter_candidates_offsets[i], filter_candidates[i], im_data,
        im_data_holder, d_cond, gpu_profiler, count_profiler);
        */
#endif
  }

  size_t ret = 0;
  if (filter_candidates[0]->GetSize() > 0 &&
      filter_candidates[1]->GetSize() > 0) {
    // apply ordering constraint among the two vertex
    uintV u0 = unmaterialized_vertices[0];
    uintV u1 = unmaterialized_vertices[1];
    CondOperator cond_operator =
        Plan::GetConditionType(u0, u1, index_based_cond);

    DeviceArray<size_t> d_total_count(1, context);
    GeneralCountTwoComputedVertices<false>(
        d_partition_id, u0, u1, cond_operator, valid_path_ids->GetSize(), NULL,
        NULL, filter_candidates_offsets[0]->GetArray(),
        filter_candidates_offsets[1]->GetArray(),
        filter_candidates[0]->GetArray(), filter_candidates[1]->GetArray(),
        d_total_count.GetArray(), context, gpu_profiler, count_profiler);

    DToH(&ret, d_total_count.GetArray(), 1);

#if defined(DEBUG)
    /*size_t ret1 = VerifyGeneralCountTwoComputedVertices<false>(
        d_partition_id, u0, u1, cond_operator, valid_path_ids->GetSize(), NULL,
        NULL, filter_candidates_offsets[0]->GetArray(),
        filter_candidates_offsets[1]->GetArray(),
        filter_candidates[0]->GetArray(), filter_candidates[1]->GetArray(),
        d_total_count_data, context, gpu_profiler, count_profiler);
    size_t h_total_count;
    DToH(&h_total_count, d_total_count_data, 1);
    assert(ret1 == h_total_count);
    */
#endif
  }

  for (size_t i = 0; i < filter_candidates_offsets.size(); ++i) {
    delete filter_candidates[i];
    filter_candidates[i] = NULL;
    delete filter_candidates_offsets[i];
    filter_candidates_offsets[i] = NULL;
  }
  return ret;
}

static size_t ItpLIGHTCount(LightItpWorkContext* wctx, size_t cur_exec_level) {
  auto dfs_id = wctx->dfs_id;
  auto context = wctx->context;
  auto plan = wctx->plan;
  auto im_data = wctx->im_data;
  auto im_data_holder = wctx->im_data_holder;

  auto& d_instances = im_data->GetInstances();
  im_data_holder->GatherImData(im_data, context);

  auto& materialized_vertices =
      plan->GetInterPartitionMaterializedVertices()[dfs_id][cur_exec_level];
  auto& unmaterialized_vertices =
      plan->GetInterPartitionComputedUnmaterializedVertices()[dfs_id]
                                                             [cur_exec_level];

  // enforce ordering for the materialized vertices
  size_t path_num = d_instances[materialized_vertices[0]]->GetSize();
  DeviceArray<size_t>* valid_path_ids = NULL;
  ItpEnforceOrderingPartialInstances(wctx, cur_exec_level, path_num,
                                     valid_path_ids);
  if (!valid_path_ids->GetSize()) {
    delete valid_path_ids;
    valid_path_ids = NULL;
    return 0;
  }

  size_t ret = 0;
  if (unmaterialized_vertices.size() == 1) {
    ret = ItpGeneralCountOneVertex(wctx, cur_exec_level, valid_path_ids);

  } else if (unmaterialized_vertices.size() == 2) {
    ret = ItpGeneralCountTwoVertices(wctx, cur_exec_level, valid_path_ids);
  } else {
    // Currently, the plan would not have more than two unmateriaalized vertices
    assert(false);
  }

  delete valid_path_ids;
  valid_path_ids = NULL;

  return ret;
}
}  // namespace Light

#endif
