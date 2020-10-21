#ifndef __EXTERNAL_GRAPH_GPU_COMMON_CUH__
#define __EXTERNAL_GRAPH_GPU_COMMON_CUH__

#include <future>
#include <thread>

#include "CPUGraph.h"
#include "Compact.cuh"
#include "CountProfiler.h"
#include "DevGraphPartition.cuh"
#include "GPUProfiler.cuh"
#include "GraphDevTracker.cuh"
#include "Iterator.cuh"
#include "LoadSubgraph.h"
#include "Transform.cuh"

// This file is to provide the common functionality and service
// to help GPUs obtain the subgraph needed.
// When the graph is too large and stored in the main memory,
// GPU need to fetch a subgraph from the main memory,
// which requires some preparation and communication between
// hosts and devices.

// Given required_vertex_ids and required_row_ptrs,
// build the global row_ptrs for the new deivce subgraph.
static void BuildSubgraphRowPtrs(size_t d_partition_id, CudaContext* context,
                                 DeviceArray<uintV>* required_vertex_ids,
                                 GraphDevTracker* graph_dev_tracker,
                                 TrackPartitionedGraph* cpu_relation,
                                 DeviceArray<uintE>*& new_row_ptrs,
                                 GPUProfiler* gpu_profiler,
                                 CountProfiler* count_profiler) {
  size_t load_vertex_count = required_vertex_ids->GetSize();
  size_t vertex_count = cpu_relation->GetVertexCount();
  uintV* required_vertex_ids_data = required_vertex_ids->GetArray();
  uintE* row_ptrs_data = graph_dev_tracker->GetGraphRowPtrs()->GetArray();

  DeviceArray<bool> bitmaps(vertex_count, context);
  bool* bitmaps_data = bitmaps.GetArray();
  CUDA_ERROR(cudaMemset(bitmaps.GetArray(), 0, sizeof(bool) * vertex_count));
  GpuUtils::Transform::Transform(
      [=] DEVICE(int index) {
        uintV v = required_vertex_ids_data[index];
        bitmaps_data[v] = true;
      },
      load_vertex_count, context);

  new_row_ptrs = new DeviceArray<uintE>(vertex_count + 1, context);
  GpuUtils::Scan::TransformScan(
      [=] DEVICE(int v) {
        return bitmaps_data[v] ? row_ptrs_data[v + 1] - row_ptrs_data[v] : 0;
      },
      vertex_count, new_row_ptrs->GetArray(),
      new_row_ptrs->GetArray() + vertex_count, context);
}

// By the required_vertex_ids, submit the task to CPUs to fetch the graph data.
// Blocking wait for the CPUs to finish the task.
// By the received graph data, build the subgraph we need in graph_partition.
static void BlockingBuildSubgraph(size_t d_partition_id, CudaContext* context,
                                  DevGraphPartition* graph_partition,
                                  DeviceArray<uintV>* required_vertex_ids,
                                  GraphDevTracker* graph_dev_tracker,
                                  TrackPartitionedGraph* cpu_relation,
                                  GPUProfiler* gpu_profiler,
                                  CountProfiler* count_profiler,
                                  size_t load_graph_thread_num) {
  gpu_profiler->StartTimer("prepare_load_graph_task_time", d_partition_id,
                           context->Stream());
  // required_row_ptrs is used by CPUs to make reading the
  // adjacent lists easier
  DeviceArray<uintE>* required_row_ptrs = NULL;
  graph_dev_tracker->BuildSubgraphRowPtrs(required_vertex_ids,
                                          required_row_ptrs, context);

  gpu_profiler->StartTimer("prepare_load_graph_task_copy_allocate_time",
                           d_partition_id, context->Stream());
  // WARNING: allocate memory ad-hoc could be slow.
  // Consider use an efficient memory allocator
  uintV* h_required_vertex_ids = new uintV[required_vertex_ids->GetSize()];
  uintE* h_required_row_ptrs = new uintE[required_row_ptrs->GetSize()];
  DToH(h_required_vertex_ids, required_vertex_ids->GetArray(),
       required_vertex_ids->GetSize());
  DToH(h_required_row_ptrs, required_row_ptrs->GetArray(),
       required_row_ptrs->GetSize());
  gpu_profiler->EndTimer("prepare_load_graph_task_copy_allocate_time",
                         d_partition_id, context->Stream());

  gpu_profiler->EndTimer("prepare_load_graph_task_time", d_partition_id,
                         context->Stream());
  count_profiler->AddCount("load_subgraph_pcie_send", d_partition_id,
                           sizeof(uintV) * required_vertex_ids->GetSize() +
                               sizeof(uintE) * required_row_ptrs->GetSize());

  // submit the task to CPUs so that they can read the adjacent lists
  size_t load_vertex_count = required_vertex_ids->GetSize();
  size_t load_edge_count = h_required_row_ptrs[load_vertex_count];
  auto load_graph_job =
      std::async(std::launch::async, LoadSubgraph, cpu_relation,
                 load_vertex_count, load_edge_count, h_required_vertex_ids,
                 h_required_row_ptrs, load_graph_thread_num);

  // clear up the old subgraph
  graph_partition->Release();

  delete required_row_ptrs;
  required_row_ptrs = NULL;

  // build row_ptrs for the new subgraph while waiting for the task
  // to be finished by CPUs
  gpu_profiler->StartTimer("build_subgraph_row_ptrs_time", d_partition_id,
                           context->Stream());
  DeviceArray<uintE>* new_row_ptrs = NULL;
  BuildSubgraphRowPtrs(d_partition_id, context, required_vertex_ids,
                       graph_dev_tracker, cpu_relation, new_row_ptrs,
                       gpu_profiler, count_profiler);
  gpu_profiler->EndTimer("build_subgraph_row_ptrs_time", d_partition_id,
                         context->Stream());

  gpu_profiler->StartTimer("wait_load_graph_time", d_partition_id,
                           context->Stream());
  // wait until it is ready
  uintV* load_graph_cols = load_graph_job.get();

  gpu_profiler->EndTimer("wait_load_graph_time", d_partition_id,
                         context->Stream());

  gpu_profiler->StartTimer("load_graph_htod_copy_edges_time", d_partition_id,
                           context->Stream());
  // load_cols is the graph data read from CPUs
  DeviceArray<uintV>* load_cols =
      new DeviceArray<uintV>(load_edge_count, context);
  HToD(load_cols->GetArray(), load_graph_cols, load_edge_count);
  gpu_profiler->EndTimer("load_graph_htod_copy_edges_time", d_partition_id,
                         context->Stream());
  count_profiler->AddCount("load_subgraph_pcie_receive", d_partition_id,
                           sizeof(uintV) * load_edge_count);

  // release
  delete[] h_required_vertex_ids;
  h_required_vertex_ids = NULL;
  delete[] h_required_row_ptrs;
  h_required_row_ptrs = NULL;
  delete[] load_graph_cols;
  load_graph_cols = NULL;

  graph_partition->Set(cpu_relation->GetVertexCount(), load_edge_count,
                       new_row_ptrs, load_cols);
}

// Inspect the adjacent lists needed to compute the candidate sets.
// This function is usually needed before loading the subgraph from main memory,
// so that we can estimate the size of the subgraph.
// BackwardNeighborGatherFunctor: given the path id, return the set of backward
// neighbors. BackwardNeighborGatherFunctor(path_id, vertices) -> vertices_count
template <typename BackwardNeighborGatherFunctor>
static void InspectRequiredAdjLists(size_t d_partition_id, CudaContext* context,
                                    size_t path_num,
                                    GraphDevTracker* graph_dev_tracker,
                                    DeviceArray<uintV>*& required_vertex_ids,
                                    BackwardNeighborGatherFunctor functor,
                                    GPUProfiler* gpu_profiler,
                                    CountProfiler* count_profiler) {
  size_t vertex_count = graph_dev_tracker->GetVertexCount();
  DeviceArray<bool> bitmaps(vertex_count, context);
  CUDA_ERROR(cudaMemset(bitmaps.GetArray(), 0, sizeof(bool) * vertex_count));
  bool* bitmaps_data = bitmaps.GetArray();

  GpuUtils::Transform::Transform(
      [=] DEVICE(int index) {
        uintV vertices[kMaxQueryVerticesNum] = {kMaxuintV};
        size_t vertices_count = functor(index, vertices);
        for (size_t i = 0; i < vertices_count; ++i) {
          uintV v = vertices[i];
          bitmaps_data[v] = true;
        }
      },
      path_num, context);

  int compact_count = 0;
  GpuUtils::Compact::Compact(GpuUtils::Iterator::CountingIterator<uintV>(0),
                             vertex_count, bitmaps.GetArray(),
                             required_vertex_ids, compact_count, context);
}

// the entry point of load subgraph from the main memory.
// The loaded subgraph is written to graph_partition.
// backward_neighbor_functor: given the path id, return the set of backward
// neighbors whose adjacent lists are needed.
template <typename BackwardNeighborGatherFunctor>
static void BlockingLoadSubgraph(
    size_t d_partition_id, CudaContext* context,
    DevGraphPartition* graph_partition, GraphDevTracker* graph_dev_tracker,
    TrackPartitionedGraph* cpu_relation,
    BackwardNeighborGatherFunctor backward_neighbor_functor, size_t path_num,
    GPUProfiler* gpu_profiler, CountProfiler* count_profiler,
    size_t load_graph_thread_num) {
  // the set of vertices whose adjacent lists are needed
  DeviceArray<uintV>* required_vertex_ids = NULL;
  gpu_profiler->StartTimer("inspect_join_time", d_partition_id,
                           context->Stream());

  InspectRequiredAdjLists(d_partition_id, context, path_num, graph_dev_tracker,
                          required_vertex_ids, backward_neighbor_functor,
                          gpu_profiler, count_profiler);
  gpu_profiler->EndTimer("inspect_join_time", d_partition_id,
                         context->Stream());

  // build subgraph
  gpu_profiler->StartTimer("build_subgraph_time", d_partition_id,
                           context->Stream());
  BlockingBuildSubgraph(d_partition_id, context, graph_partition,
                        required_vertex_ids, graph_dev_tracker, cpu_relation,
                        gpu_profiler, count_profiler, load_graph_thread_num);
  gpu_profiler->EndTimer("build_subgraph_time", d_partition_id,
                         context->Stream());
  delete required_vertex_ids;
  required_vertex_ids = NULL;
}

#endif
