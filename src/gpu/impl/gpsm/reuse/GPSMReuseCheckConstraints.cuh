#ifndef __HYBRID_GPSM_REUSE_COMPONENT_GPU_CHECK_CONSTRAINTS_CUH__
#define __HYBRID_GPSM_REUSE_COMPONENT_GPU_CHECK_CONSTRAINTS_CUH__

#include <cub/cub.cuh>
#include "CheckConstraintsCommon.cuh"
#include "CheckConstraintsReuseCommon.cuh"
#include "Copy.cuh"
#include "CountProfiler.h"
#include "CudaContext.cuh"
#include "DevReuseTraversalPlan.cuh"
#include "GPSMReuseBinSearch.cuh"
#include "GPSMReuseCommon.cuh"
#include "GPUFilter.cuh"
#include "GPUTimer.cuh"
#include "Scan.cuh"
#include "Task.h"
#include "Transform.cuh"

namespace GpsmReuse {

static void FindCacheOffsets(GpsmReuseWorkContext* wctx, uintV cur_level,
                             BatchSpec* batch_spec, bool* reuse_valid_data,
                             size_t** d_seq_reuse_indices_data) {
  auto context = wctx->context;
  auto im_data = wctx->im_data;
  auto im_data_holder = wctx->im_data_holder;
  auto cache_data = wctx->cache_data;
  DevReuseTraversalPlan* dev_plan =
      static_cast<DevReuseTraversalPlan*>(wctx->dev_plan);

  auto& d_inst = im_data->GetInst();
  auto d_seq_inst_data = im_data_holder->GetSeqInst()->GetArray();

  auto d_seq_inst_next_offsets_data =
      cache_data->GetSeqInstNextOffsets()->GetArray();
  auto d_seq_inst_parents_indices_data =
      cache_data->GetSeqInstParentsIndices()->GetArray();
  auto d_seq_cache_next_offsets_data =
      cache_data->GetSeqCacheNextOffsets()->GetArray();
  auto d_seq_cache_instances_data =
      cache_data->GetSeqCacheInstances()->GetArray();
  // the instances represented in tree structure
  auto d_seq_tree_instances_data = cache_data->GetSeqInstPtrs()->GetArray();

  auto d_seq_level_batch_offsets_start_data =
      cache_data->GetSeqLevelBatchOffsetsStart()->GetArray();
  auto d_seq_level_batch_offsets_end_data =
      cache_data->GetSeqLevelBatchOffsetsEnd()->GetArray();

  size_t path_num = d_inst[cur_level - 1]->GetSize();
  size_t batch_offset_left = batch_spec->GetBatchLeftEnd();

  DevVertexReuseIntersectPlan* level_reuse_intersect_plan_data =
      dev_plan->GetLevelReuseIntersectPlan()->GetArray();

  GpuUtils::Transform::Transform(
      [=] DEVICE(int index) {
        uintV M[kMaxQueryVerticesNum];
        for (size_t i = 0; i < cur_level; ++i) {
          M[i] = d_seq_inst_data[i][index];
        }

        size_t last_level_index = batch_offset_left + index;
        bool valid = ThreadFindCachedOffset(
            level_reuse_intersect_plan_data[cur_level], M, last_level_index,
            cur_level - 1, d_seq_tree_instances_data,
            d_seq_inst_parents_indices_data, d_seq_inst_next_offsets_data,
            d_seq_level_batch_offsets_start_data,
            d_seq_level_batch_offsets_end_data, index,
            d_seq_reuse_indices_data);

        reuse_valid_data[index] = valid;
      },
      path_num, context);

#if defined(REUSE_PROFILE)
  auto d_partition_id = wctx->d_partition_id;
  auto count_profiler = wctx->count_profiler;
  DeviceArray<size_t> d_total_reuse_count(1, context);
  GpuUtils::Reduce::TransformReduce(
      [=] DEVICE(int index) {
        return reuse_valid_data[index] ? (size_t)1 : (size_t)0;
      },
      path_num, d_total_reuse_count.GetArray(), context);
  size_t h_total_reuse_count;
  DToH(&h_total_reuse_count, d_total_reuse_count.GetArray(), 1);
  count_profiler->AddCount("reuse_count", d_partition_id, h_total_reuse_count);
  count_profiler->AddCount("intersect_count", d_partition_id, path_num);
#endif
}

static void Materialize(GpsmReuseWorkContext* wctx, uintV cur_level,
                        BatchSpec* batch_spec,
                        DeviceArray<size_t>* overall_children_offset,
                        DeviceArray<uintV>* overall_children,
                        DeviceArray<size_t>* intersect_children_offset,
                        DeviceArray<uintV>* intersect_children,
                        bool cache_required) {
  auto d_partition_id = wctx->d_partition_id;
  auto context = wctx->context;
  auto im_data = wctx->im_data;
  auto cache_data = wctx->cache_data;
  auto plan = wctx->plan;
  auto gpu_profiler = wctx->gpu_profiler;

  auto& d_inst = im_data->GetInst();
  size_t path_num = d_inst[cur_level - 1]->GetSize();

  gpu_profiler->StartTimer("materialize_time", d_partition_id,
                           context->Stream());

  std::swap(d_inst[cur_level], overall_children);
  ReleaseIfExists(overall_children);

  size_t total_overall_children_count = d_inst[cur_level]->GetSize();
  DeviceArray<size_t>* parent_indices =
      new DeviceArray<size_t>(total_overall_children_count, context);
  GpuUtils::LoadBalance::LoadBalanceSearch<MGPULaunchBoxVT1>(
      total_overall_children_count, overall_children_offset->GetArray(),
      path_num, parent_indices->GetArray(), context);
  for (size_t i = 0; i < cur_level; ++i) {
    DeviceArray<uintV>* output =
        new DeviceArray<uintV>(total_overall_children_count, context);
    GpuUtils::Copy::Gather(parent_indices->GetArray(),
                           total_overall_children_count, d_inst[i]->GetArray(),
                           output->GetArray(), context);
    std::swap(d_inst[i], output);
    ReleaseIfExists(output);
  }

  if (cur_level == plan->GetVertexCount() - 1) {
    delete overall_children_offset;
    overall_children_offset = NULL;
    delete parent_indices;
    parent_indices = NULL;
    assert(!cache_required);
  } else {
    auto& d_inst_next_offsets = cache_data->GetInstNextOffsets();
    auto& d_inst_parents_indices = cache_data->GetInstParentsIndices();
    auto& d_inst_ptrs = cache_data->GetInstPtrs();
    auto& d_cache_next_offsets = cache_data->GetCacheNextOffsets();
    auto& d_cache_instances = cache_data->GetCacheInstances();
    // take care of batch offset
    GpuUtils::Transform::Apply<ADD>(parent_indices->GetArray(),
                                    parent_indices->GetSize(),
                                    batch_spec->GetBatchLeftEnd(), context);

    d_inst_next_offsets[cur_level - 1] = overall_children_offset;
    d_inst_parents_indices[cur_level] = parent_indices;
    d_inst_ptrs[cur_level] = d_inst[cur_level]->GetArray();
    cache_data->SetLevelBatchoffset(cur_level - 1,
                                    batch_spec->GetBatchLeftEnd(),
                                    batch_spec->GetBatchRightEnd());

    if (cache_required) {
      d_cache_next_offsets[cur_level - 1] = intersect_children_offset;
      d_cache_instances[cur_level] = intersect_children;
    } else {
      d_cache_next_offsets[cur_level - 1] = NULL;
      d_cache_instances[cur_level] = NULL;
    }
  }

  gpu_profiler->EndTimer("materialize_time", d_partition_id, context->Stream());
}
static void JoinPhase(GpsmReuseWorkContext* wctx, uintV cur_level,
                      BatchSpec* batch_spec) {
  auto d_partition_id = wctx->d_partition_id;
  auto context = wctx->context;
  auto im_data = wctx->im_data;
  auto im_data_holder = wctx->im_data_holder;
  auto cache_data = wctx->cache_data;
  auto graph_partition = wctx->graph_partition;
  ReuseTraversalPlan* plan = static_cast<ReuseTraversalPlan*>(wctx->plan);
  DevReuseTraversalPlan* dev_plan =
      static_cast<DevReuseTraversalPlan*>(wctx->dev_plan);
  auto gpu_profiler = wctx->gpu_profiler;

  /// ========== prepare ==========
  gpu_profiler->StartTimer("join_phase_prepare_time", d_partition_id,
                           context->Stream());
  assert(cur_level > 0);
  // im_data
  im_data_holder->GatherImData(im_data, context);
  auto& d_inst = im_data->GetInst();
  size_t path_num = d_inst[cur_level - 1]->GetSize();

  // cache_data
  cache_data->GatherCacheData(context);
  auto d_seq_inst_data = im_data_holder->GetSeqInst()->GetArray();
  size_t** d_seq_cache_next_offsets_data =
      cache_data->GetSeqCacheNextOffsets()->GetArray();
  uintV** d_seq_cache_instances_data =
      cache_data->GetSeqCacheInstances()->GetArray();

  // reuse_valid, reuse_indices_vec, d_seq_reuse_indices
  DeviceArray<bool>* reuse_valid = new DeviceArray<bool>(path_num, context);
  bool* reuse_valid_data = reuse_valid->GetArray();

  std::vector<DeviceArray<size_t>*> reuse_indices_vec;
  auto& reuse_conn_meta =
      plan->GetLevelReuseIntersectPlan()[cur_level].GetReuseConnectivityMeta();
  for (size_t i = 0; i < reuse_conn_meta.size(); ++i) {
    reuse_indices_vec.push_back(new DeviceArray<size_t>(path_num, context));
  }
  DeviceArray<size_t*>* d_seq_reuse_indices = NULL;
  BuildTwoDimensionDeviceArray(d_seq_reuse_indices, &reuse_indices_vec,
                               context);
  size_t** d_seq_reuse_indices_data = d_seq_reuse_indices->GetArray();

  // graph_partition
  uintE* row_ptrs = graph_partition->GetRowPtrs()->GetArray();
  uintV* cols = graph_partition->GetCols()->GetArray();
  size_t vertex_count = graph_partition->GetVertexCount();

  // plan
  bool cache_required = plan->GetCacheRequired(cur_level);
  DevVertexReuseIntersectPlan* level_reuse_intersect_plan_data =
      dev_plan->GetLevelReuseIntersectPlan()->GetArray() + cur_level;
  auto conn = dev_plan->GetBackwardConnectivity()->GetArray() + cur_level;
  auto cond = dev_plan->GetBackwardCondition()->GetArray() + cur_level;

  gpu_profiler->EndTimer("join_phase_prepare_time", d_partition_id,
                         context->Stream());

  // ====== choose pivot levels =======
  // find the position of cached result if possible
  gpu_profiler->StartTimer("reuse_find_cache_offsets", d_partition_id,
                           context->Stream());
  FindCacheOffsets(wctx, cur_level, batch_spec, reuse_valid_data,
                   d_seq_reuse_indices_data);
  gpu_profiler->EndTimer("reuse_find_cache_offsets", d_partition_id,
                         context->Stream());

  // ====== functor ========
  auto warp_prepare_functor =
      [=] DEVICE(uintV * M, size_t path_id, bool& shm_pivot_from_cache,
                 uintV& shm_pivot_index, uintV& shm_pivot_level) {
        for (uintV i = 0; i < cur_level; ++i) {
          M[i] = d_seq_inst_data[i][path_id];
        }
        if (reuse_valid_data[path_id]) {
          ThreadChoosePivotIndexReuse(*level_reuse_intersect_plan_data, M,
                                      path_id, d_seq_reuse_indices_data,
                                      d_seq_cache_next_offsets_data, row_ptrs,
                                      shm_pivot_from_cache, shm_pivot_index);
        } else {
          shm_pivot_level = ThreadChoosePivotLevel(*conn, M, row_ptrs);
        }
      };

  auto gen_candidates_functor = [=] DEVICE(uintV * M, size_t path_id,
                                           bool shm_pivot_from_cache,
                                           uintV shm_pivot_index,
                                           uintV shm_pivot_level,
                                           uintV*& candidates,
                                           size_t& candidates_count) {
    if (reuse_valid_data[path_id]) {
      if (shm_pivot_from_cache) {
        auto& reuse_conn_meta =
            level_reuse_intersect_plan_data->GetReuseConnectivityMeta(
                shm_pivot_index);
        uintV target_level = reuse_conn_meta.GetSourceVertex();
        size_t cached_offset =
            d_seq_reuse_indices_data[shm_pivot_index][path_id];
        candidates =
            d_seq_cache_instances_data[target_level] +
            d_seq_cache_next_offsets_data[target_level - 1][cached_offset];
        candidates_count =
            d_seq_cache_next_offsets_data[target_level - 1][cached_offset + 1] -
            d_seq_cache_next_offsets_data[target_level - 1][cached_offset];
      } else {
        auto& separate_conn =
            level_reuse_intersect_plan_data->GetSeparateConnectivity();
        uintV pivot_level = separate_conn.Get(shm_pivot_index);
        uintV pivot_vertex = M[pivot_level];
        candidates = cols + row_ptrs[pivot_vertex];
        candidates_count = row_ptrs[pivot_vertex + 1] - row_ptrs[pivot_vertex];
      }
    } else {
      uintV pivot_vertex = M[shm_pivot_level];
      candidates = cols + row_ptrs[pivot_vertex];
      candidates_count = row_ptrs[pivot_vertex + 1] - row_ptrs[pivot_vertex];
    }
  };

  auto verify_functor =
      [=] DEVICE(uintV * M, size_t path_id, bool shm_pivot_from_cache,
                 uintV shm_pivot_index, uintV shm_pivot_level, uintV candidate,
                 bool& intersect_valid, bool& valid) {
        if (reuse_valid_data[path_id]) {
          intersect_valid = ThreadCheckConnectivityReuseOpt(
              *level_reuse_intersect_plan_data, M, path_id, candidate,
              d_seq_reuse_indices_data, d_seq_cache_next_offsets_data,
              d_seq_cache_instances_data, row_ptrs, cols, shm_pivot_from_cache,
              shm_pivot_index);
        } else {
          intersect_valid = ThreadCheckConnectivity(
              *conn, M, candidate, shm_pivot_level, row_ptrs, cols);
        }

        valid = intersect_valid;
        if (valid) {
          valid = ThreadCheckCondition(*cond, M, candidate);
        }
      };

  // ========== intersect ============
  DeviceArray<size_t>*overall_children_offset = NULL,
  *intersect_children_offset = NULL;
  DeviceArray<uintV>*overall_children = NULL, *intersect_children = NULL;

  gpu_profiler->StartTimer("check_constraints_time", d_partition_id,
                           context->Stream());

  GpuUtils::Intersect::GpsmReuseBinSearch(
      warp_prepare_functor, gen_candidates_functor, verify_functor, path_num,
      overall_children_offset, overall_children, intersect_children_offset,
      intersect_children, context, cache_required);
  gpu_profiler->EndTimer("check_constraints_time", d_partition_id,
                         context->Stream());

  // release
  delete reuse_valid;
  reuse_valid = NULL;
  for (size_t i = 0; i < reuse_indices_vec.size(); ++i) {
    delete reuse_indices_vec[i];
    reuse_indices_vec[i] = NULL;
  }
  reuse_indices_vec.clear();
  delete d_seq_reuse_indices;
  d_seq_reuse_indices = NULL;

  // ======= materialize =======
  Materialize(wctx, cur_level, batch_spec, overall_children_offset,
              overall_children, intersect_children_offset, intersect_children,
              cache_required);
}
}  // namespace GpsmReuse

#endif
