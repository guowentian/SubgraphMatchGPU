#ifndef __GPU_IMPLEMENTATION_COMMON_EXTERNAL_BATCH_MANAGER_CUH__
#define __GPU_IMPLEMENTATION_COMMON_EXTERNAL_BATCH_MANAGER_CUH__

#include "BatchManager.cuh"
#include "CPUGraph.h"
#include "CountProfiler.h"
#include "DevGraphPartition.cuh"
#include "GPUProfiler.cuh"
#include "GraphDevTracker.cuh"

#include "Reduce.cuh"
#include "Transform.cuh"

// This file provide facilities to manage the batches when
// we should load the graph from the main memory into GPUs.
// In this time, the batch management should estimate the size of the subgraph
// loaded from the main memory to avoid device memory overflow.

// Estimate the subgraph size for a range of paths.
// BackwardNeighborFunctor: given the path id, return the set of
// vertices that are backward neighbors and whose adjacent lists are required.
// BackwardNeighborFunctor(index, vertices) -> vertices_count
template <typename BackwardNeighborFunctor>
static size_t EstimateSubgraphSize(size_t d_partition_id, CudaContext *context,
                                   uintE *row_ptrs, size_t vertex_count,
                                   size_t batch_left_first, size_t path_num,
                                   BackwardNeighborFunctor func) {
  DeviceArray<bool> bitmaps(vertex_count, context);
  bool *bitmaps_data = bitmaps.GetArray();
  CUDA_ERROR(cudaMemset(bitmaps_data, 0, sizeof(bool) * vertex_count));
  GpuUtils::Transform::Transform(
      [=] DEVICE(int index) {
        uintV vertices[kMaxQueryVerticesNum] = {kMaxuintV};
        size_t vertices_count = func(batch_left_first + index, vertices);
        for (size_t i = 0; i < vertices_count; ++i) {
          uintV v = vertices[i];
          bitmaps_data[v] = true;
        }
      },
      path_num, context);

  DeviceArray<size_t> reduction(1, context);
  GpuUtils::Reduce::TransformReduce(
      [=] DEVICE(int index) {
        return bitmaps_data[index] ? row_ptrs[index + 1] - row_ptrs[index] : 0;
      },
      vertex_count, reduction.GetArray(), context);

  size_t ret;
  DToH(&ret, reduction.GetArray(), 1);
  return ret;
}

// Given the currently achieved batch_manager, which is based on a pessmistic
// estimation of memory cost for each path,
// combine the batches if possible to reduce #batches.
// To combine the batches, we use fine-grained calculation the subgraph size
// for a number of batches, and combine the batches if the size do not exceed
// the batch size.
// BackwardNeighborFunctor: given the path id, return the set of
// vertices that are backward neighbors and whose adjacent lists are required.
// BackwardNeighborFunctor(index, vertices) -> vertices_count
template <typename BackwardNeighborFunctor>
static void CombineLoadSubgraphBatch(
    size_t d_partition_id, CudaContext *context, BackwardNeighborFunctor func,
    GraphDevTracker *graph_dev_tracker, BatchManager *batch_manager,
    size_t parent_factor, size_t children_factor,
    DeviceArray<size_t> *children_count) {
  uintE *row_ptrs = graph_dev_tracker->GetGraphRowPtrs()->GetArray();
  size_t *children_count_data = children_count->GetArray();
  size_t vertex_count = graph_dev_tracker->GetVertexCount();

  std::vector<BatchSpec> new_batches;

  size_t cur_batch_id = 0;
  while (cur_batch_id < batch_manager->GetBatchNum()) {
    size_t step = 1;
    size_t batch_right_range = cur_batch_id + step;

    while (cur_batch_id + step <= batch_manager->GetBatchNum()) {
      // test whether [cur_batch_id, cur_batch_id+step)
      // can be comebined.

      size_t batch_left_first =
          batch_manager->GetBatch(cur_batch_id).GetBatchLeftEnd();
      size_t batch_right_first =
          batch_manager->GetBatch(cur_batch_id + step - 1).GetBatchRightEnd();
      BatchSpec batch_spec(batch_left_first, batch_right_first);

      // calculate the subgraph size in the range
      size_t subgraph_size = EstimateSubgraphSize(
          d_partition_id, context, row_ptrs, vertex_count, batch_left_first,
          batch_spec.GetBatchCount(), func);

      // calculate the temporary memory cost in this range
      DeviceArray<size_t> d_temporary_memory_cost(1, context);
      GpuUtils::Reduce::TransformReduce(
          [=] DEVICE(int index) {
            return children_count_data[batch_left_first + index] *
                       children_factor +
                   parent_factor;
          },
          batch_spec.GetBatchCount(), d_temporary_memory_cost.GetArray(),
          context);
      size_t temporary_memory_cost;
      DToH(&temporary_memory_cost, d_temporary_memory_cost.GetArray(), 1);

      // if the total cost is within range, continue to increase step
      // otherwise, no more valid results, break
      size_t total_memory_cost =
          subgraph_size * sizeof(uintV) + temporary_memory_cost;
      size_t batch_size = batch_manager->GetBatchSize();
      if (total_memory_cost <= batch_size) {
        batch_right_range = cur_batch_id + step;
        step = step * 2;
      } else {
        if (step > 1) step /= 2;
        break;
      }
    }

    // add the new batch: [cur_batch_id, batch_right_range)
    assert(cur_batch_id < batch_right_range);
    size_t batch_left_first =
        batch_manager->GetBatch(cur_batch_id).GetBatchLeftEnd();
    size_t batch_right_first =
        batch_manager->GetBatch(batch_right_range - 1).GetBatchRightEnd();
    new_batches.push_back(BatchSpec(batch_left_first, batch_right_first));

    cur_batch_id = batch_right_range;
  }

  // assign the new combined batches into batch_manager
  batch_manager->SetBatches(new_batches);
}

#endif