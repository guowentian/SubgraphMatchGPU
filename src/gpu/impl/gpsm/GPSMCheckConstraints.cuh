#ifndef __HYBRID_GPSM_PIPELINE_GPU_CHECK_CONSTRAINTS_CUH__
#define __HYBRID_GPSM_PIPELINE_GPU_CHECK_CONSTRAINTS_CUH__

#include "CheckConstraintsCommon.cuh"
#include "Copy.cuh"
#include "CountProfiler.h"
#include "CudaContext.cuh"
#include "DevTraversalPlan.cuh"
#include "GPSMCommon.cuh"
#include "GPUFilter.cuh"
#include "GPUTimer.cuh"
#include "Intersect.cuh"
#include "LoadBalance.cuh"
#include "Scan.cuh"
#include "Task.h"
#include "Transform.cuh"
#include "TraversalPlan.h"

namespace Gpsm {

static void InitFirstLevel(GpsmWorkContext* wctx, uintV cur_level,
                           BatchSpec* batch_spec, IntraPartTask* task) {
  auto d_partition_id = wctx->d_partition_id;
  auto context = wctx->context;
  auto im_data = wctx->im_data;
  auto gpu_profiler = wctx->gpu_profiler;

  auto& d_inst = im_data->GetInst();
  assert(cur_level == 0);
  gpu_profiler->StartTimer("materialize_time", d_partition_id,
                           context->Stream());
  size_t path_num = batch_spec->GetBatchCount();
  ReAllocate(d_inst[cur_level], path_num, context);
  uintV start_vertex_id = task->start_offset_ + batch_spec->GetBatchLeftEnd();
  GpuUtils::Transform::Sequence<uintV>(d_inst[cur_level]->GetArray(), path_num,
                                       start_vertex_id, context);
  gpu_profiler->EndTimer("materialize_time", d_partition_id, context->Stream());
}

static void JoinPhase(GpsmWorkContext* wctx, uintV cur_level) {
  auto d_partition_id = wctx->d_partition_id;
  auto context = wctx->context;
  auto im_data = wctx->im_data;
  auto im_data_holder = wctx->im_data_holder;
  auto graph_partition = wctx->graph_partition;
  auto dev_plan = wctx->dev_plan;
  auto gpu_profiler = wctx->gpu_profiler;

  im_data_holder->GatherImData(im_data, context);
  auto& d_inst = im_data->GetInst();
  auto d_seq_inst_data = im_data_holder->GetSeqInst()->GetArray();

  assert(cur_level > 0);
  size_t path_num = d_inst[cur_level - 1]->GetSize();
  uintE* row_ptrs = graph_partition->GetRowPtrs()->GetArray();
  uintV* cols = graph_partition->GetCols()->GetArray();
  auto backward_conn =
      dev_plan->GetBackwardConnectivity()->GetArray() + cur_level;
  auto backward_cond = dev_plan->GetBackwardCondition()->GetArray() + cur_level;

  auto instance_gather_functor = [=] DEVICE(int index, uintV* M) {
    for (int i = 0; i < cur_level; ++i) {
      M[i] = d_seq_inst_data[i][index];
    }
  };

  DeviceArray<size_t>* children_offsets = NULL;
  DeviceArray<uintV>* children = NULL;
  gpu_profiler->StartTimer("check_constraints_time", d_partition_id,
                           context->Stream());
  GpuUtils::Intersect::Intersect<
      GpuUtils::Intersect::ProcessMethod::GPSM_BIN_SEARCH, true, false>(
      instance_gather_functor, path_num, row_ptrs, cols, backward_conn,
      backward_cond, (uintP*)NULL, (uintV)0, (uintV)1, children_offsets,
      children, context);
  gpu_profiler->EndTimer("check_constraints_time", d_partition_id,
                         context->Stream());

  d_inst[cur_level] = children;

  // materialize into d_inst
  gpu_profiler->StartTimer("materialize_time", d_partition_id,
                           context->Stream());
  size_t total_children_count = children->GetSize();
  DeviceArray<size_t>* parents_indices =
      new DeviceArray<size_t>(total_children_count, context);
  GpuUtils::LoadBalance::LoadBalanceSearch<MGPULaunchBoxVT1>(
      total_children_count, children_offsets->GetArray(), path_num,
      parents_indices->GetArray(), context);
  // expand previous levels
  for (size_t i = 0; i < cur_level; ++i) {
    DeviceArray<uintV>* output =
        new DeviceArray<uintV>(total_children_count, context);
    GpuUtils::Copy::Gather(parents_indices->GetArray(), total_children_count,
                           d_inst[i]->GetArray(), output->GetArray(), context);
    std::swap(d_inst[i], output);
    ReleaseIfExists(output);
  }
  delete parents_indices;
  parents_indices = NULL;
  gpu_profiler->EndTimer("materialize_time", d_partition_id, context->Stream());

  delete children_offsets;
  children_offsets = NULL;
}
}  // namespace Gpsm

#endif
