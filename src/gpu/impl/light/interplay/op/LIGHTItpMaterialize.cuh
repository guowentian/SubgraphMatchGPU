#ifndef __HYBRID_LIGHT_INTERPLAY_PIPELINE_GPU_COMPONENT_MATERIALIZE_CUH__
#define __HYBRID_LIGHT_INTERPLAY_PIPELINE_GPU_COMPONENT_MATERIALIZE_CUH__

#include "LIGHTMaterialize.cuh"

#include "LoadBalance.cuh"

namespace Light {
static void InitFirstTwoLevel(LightWorkContext* wctx,
                              TrackPartitionedGraph* cpu_graph,
                              InterPartTask* task, uintV u0, uintV u1,
                              BatchSpec* batch_spec) {
  auto context = wctx->context;
  auto im_data = wctx->im_data;
  auto dev_plan = wctx->dev_plan;
  auto& d_instances = im_data->GetInstances();

  // decide the range [inter_part_start_vertex_id, inter_part_end_vertex_id]
  size_t batch_inter_part_edge_start_offset =
      task->start_offset_ + batch_spec->GetBatchLeftEnd();
  size_t batch_inter_part_edge_end_offset =
      task->start_offset_ + batch_spec->GetBatchRightEnd();
  size_t path_num = batch_spec->GetBatchCount();

  uintE* h_inter_part_row_ptrs = cpu_graph->GetInterRowPtrs();
  uintV* h_inter_part_cols = cpu_graph->GetInterCols();
  size_t vertex_count = cpu_graph->GetVertexCount();
  uintV inter_part_start_vertex_id =
      std::upper_bound(h_inter_part_row_ptrs,
                       h_inter_part_row_ptrs + vertex_count + 1,
                       batch_inter_part_edge_start_offset) -
      h_inter_part_row_ptrs;
  --inter_part_start_vertex_id;
  uintV inter_part_end_vertex_id =
      std::upper_bound(h_inter_part_row_ptrs,
                       h_inter_part_row_ptrs + vertex_count + 1,
                       batch_inter_part_edge_end_offset) -
      h_inter_part_row_ptrs;
  --inter_part_end_vertex_id;
  assert(inter_part_start_vertex_id <= inter_part_end_vertex_id);
  assert(inter_part_end_vertex_id < vertex_count);

  // children_offsets
  size_t segments_num =
      inter_part_end_vertex_id - inter_part_start_vertex_id + 1;
  DeviceArray<size_t> children_offsets(segments_num + 1, context);
  DeviceArray<uintE>* d_inter_part_row_ptrs =
      new DeviceArray<uintE>(segments_num + 1, context);
  HToD(d_inter_part_row_ptrs->GetArray(),
       h_inter_part_row_ptrs + inter_part_start_vertex_id, segments_num + 1);
  GpuUtils::Transform::Apply<ASSIGNMENT>(children_offsets.GetArray(),
                                         d_inter_part_row_ptrs->GetArray(),
                                         segments_num + 1, context);
  delete d_inter_part_row_ptrs;
  d_inter_part_row_ptrs = NULL;

  CUDA_ERROR(cudaMemset(children_offsets.GetArray(), 0, sizeof(size_t)));
  GpuUtils::Transform::Apply<MINUS>(
      children_offsets.GetArray() + 1, segments_num,
      batch_inter_part_edge_start_offset, context);
  HToD(children_offsets.GetArray() + segments_num, &path_num, 1);

  // init d_instances[u0] and d_instances[u1]
  ReAllocate(d_instances[u0], path_num, context);
  ReAllocate(d_instances[u1], path_num, context);

  GpuUtils::LoadBalance::LoadBalanceSearch<MGPULaunchBoxVT1>(
      path_num, children_offsets.GetArray(), segments_num,
      d_instances[u0]->GetArray(), context);
  GpuUtils::Transform::Apply<ADD>(d_instances[u0]->GetArray(), path_num,
                                  inter_part_start_vertex_id, context);

  HToD(d_instances[u1]->GetArray(),
       h_inter_part_cols + batch_inter_part_edge_start_offset, path_num,
       context->Stream());

  // verify condition
  auto cond = dev_plan->GetComputedOrdering()->GetArray() + u1;
  uintV* d_instances0_data = d_instances[u0]->GetArray();
  uintV* d_instances1_data = d_instances[u1]->GetArray();
  DeviceArray<bool> bitmaps(path_num, context);
  bool* bitmaps_data = bitmaps.GetArray();

  GpuUtils::Transform::Transform(
      [=] DEVICE(int index) {
        uintV v0 = d_instances0_data[index];
        uintV v1 = d_instances1_data[index];
#if defined(DEBUG)
        assert(cond->GetCount() == 1 && cond->Get(0).GetOperand() == u0);
#endif
        bitmaps_data[index] = GpuUtils::Filter::CheckVertexCondition(
            v1, v0, cond->Get(0).GetOperator());
      },
      path_num, context);

  // compact
  int compact_count = 0;
  DeviceArray<uintV>* compact_instances0 = NULL;
  GpuUtils::Compact::Compact(d_instances[u0], path_num, bitmaps_data,
                             compact_instances0, compact_count, context);
  std::swap(d_instances[u0], compact_instances0);
  ReleaseIfExists(compact_instances0);

  compact_count = 0;
  DeviceArray<uintV>* compact_instances1 = NULL;
  GpuUtils::Compact::Compact(d_instances[u1], path_num, bitmaps_data,
                             compact_instances1, compact_count, context);
  std::swap(d_instances[u1], compact_instances1);
  ReleaseIfExists(compact_instances1);
}

}  // namespace Light

#endif
