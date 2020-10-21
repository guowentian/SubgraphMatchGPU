#ifndef __EXTERNAL_INCREMENTAL_GRAPH_GPU_COMMON_CUH__
#define __EXTERNAL_INCREMENTAL_GRAPH_GPU_COMMON_CUH__

#include "EXTGraphGPUCommon.cuh"

#include "LoadBalance.cuh"

// Inspect the set of vertices whose adjacnet lists are required to be loaded,
// considering the current subgraph in the GPU memory.
// If the corresponding adjacent lists are in GPu memory already, don't need to
// load.
// required_vertex_ids: all the vertices whose adjacent lists are needed.
// toload_vertex_ids: the subset of required_vertex_ids that are not in GPU
// memory.
// ingpu_vertex_ids: the subset of required_vertex_ids that are in GPU
// memory already.
// BackwardNeighborGatherFunctor: given the path id,
// find the set of backward neighbor vertices whose adjacent lists are needed.
template <typename BackwardNeighborGatherFunctor>
static void IncInspectRequiredAdjLists(
    size_t d_partition_id, CudaContext* context,
    DevGraphPartition* graph_partition, GraphDevTracker* graph_dev_tracker,
    DeviceArray<uintV>*& required_vertex_ids,
    DeviceArray<uintV>*& toload_vertex_ids,
    DeviceArray<uintV>*& ingpu_vertex_ids,
    BackwardNeighborGatherFunctor functor, size_t path_num,
    GPUProfiler* gpu_profiler, CountProfiler* count_profiler) {
  uintE* global_row_ptrs = graph_dev_tracker->GetGraphRowPtrs()->GetArray();
  uintE* cur_row_ptrs = graph_partition->GetRowPtrs()->GetArray();
  size_t vertex_count = graph_dev_tracker->GetVertexCount();

  DeviceArray<bool> required_bitmaps(vertex_count, context);
  DeviceArray<bool> toload_bitmaps(vertex_count, context);
  DeviceArray<bool> ingpu_bitmaps(vertex_count, context);
  CUDA_ERROR(
      cudaMemset(required_bitmaps.GetArray(), 0, sizeof(bool) * vertex_count));
  CUDA_ERROR(
      cudaMemset(toload_bitmaps.GetArray(), 0, sizeof(bool) * vertex_count));
  CUDA_ERROR(
      cudaMemset(ingpu_bitmaps.GetArray(), 0, sizeof(bool) * vertex_count));
  bool* required_bitmaps_data = required_bitmaps.GetArray();
  bool* toload_bitmaps_data = toload_bitmaps.GetArray();
  bool* ingpu_bitmaps_data = ingpu_bitmaps.GetArray();

  GpuUtils::Transform::Transform(
      [=] DEVICE(int index) {
        uintV vertices[kMaxQueryVerticesNum] = {kMaxuintV};
        size_t vertices_count = functor(index, vertices);
        for (size_t i = 0; i < vertices_count; ++i) {
          uintV v = vertices[i];
#if defined(DEBUG)
          assert(v < vertex_count);
#endif
          required_bitmaps_data[v] = true;

          size_t cur_count = cur_row_ptrs[v + 1] - cur_row_ptrs[v];
          // currently not in device memory
          if (cur_count == 0) {
            toload_bitmaps_data[v] = true;
          } else {
            ingpu_bitmaps_data[v] = true;
          }
        }
      },
      path_num, context);

  int compact_count = 0;
  GpuUtils::Compact::Compact(GpuUtils::Iterator::CountingIterator<uintV>(0),
                             vertex_count, required_bitmaps_data,
                             required_vertex_ids, compact_count, context);
  GpuUtils::Compact::Compact(GpuUtils::Iterator::CountingIterator<uintV>(0),
                             vertex_count, toload_bitmaps_data,
                             toload_vertex_ids, compact_count, context);
  GpuUtils::Compact::Compact(GpuUtils::Iterator::CountingIterator<uintV>(0),
                             vertex_count, ingpu_bitmaps_data, ingpu_vertex_ids,
                             compact_count, context);
}

// Build row_ptrs for the new subgraph
static void BuildSubgraphRowPtrs(size_t d_partition_id, CudaContext* context,
                                 DeviceArray<uintV>* required_vertex_ids,
                                 GraphDevTracker* graph_dev_tracker,
                                 DeviceArray<uintE>*& new_row_ptrs,
                                 GPUProfiler* gpu_profiler,
                                 CountProfiler* count_profiler) {
  size_t load_vertex_count = required_vertex_ids->GetSize();
  size_t vertex_count = graph_dev_tracker->GetVertexCount();
  uintV* required_vertex_ids_data = required_vertex_ids->GetArray();
  uintE* global_row_ptrs = graph_dev_tracker->GetGraphRowPtrs()->GetArray();

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
        return bitmaps_data[v] ? global_row_ptrs[v + 1] - global_row_ptrs[v]
                               : 0;
      },
      vertex_count, new_row_ptrs->GetArray(),
      new_row_ptrs->GetArray() + vertex_count, context);
}

// From the subgraph in GPU (graph_partition), load the adjcent lists
// of those vertices that are required in the new subgraph
static void LoadAdjListsInGPU(size_t d_partition_id, CudaContext* context,
                              DevGraphPartition* graph_partition,
                              DeviceArray<uintV>* ingpu_vertex_ids,
                              DeviceArray<uintE>* new_row_ptrs,
                              DeviceArray<uintV>* new_cols,
                              GPUProfiler* gpu_profiler,
                              CountProfiler* count_profiler) {
  size_t vertex_count = graph_partition->GetVertexCount();
  size_t ingpu_vertex_count = ingpu_vertex_ids->GetSize();
  if (ingpu_vertex_count == 0) {
    return;
  }
  uintV* ingpu_vertex_ids_data = ingpu_vertex_ids->GetArray();
  uintE* cur_row_ptrs = graph_partition->GetRowPtrs()->GetArray();
  uintV* cur_cols = graph_partition->GetCols()->GetArray();
  uintE* new_row_ptrs_data = new_row_ptrs->GetArray();
  uintV* new_cols_data = new_cols->GetArray();

  DeviceArray<size_t> workload_offsets(ingpu_vertex_count + 1, context);
  GpuUtils::Scan::TransformScan(
      [=] DEVICE(int index) {
        uintV v = ingpu_vertex_ids_data[index];

        size_t count = cur_row_ptrs[v + 1] - cur_row_ptrs[v];
#if defined(DEBUG)
        assert(v < vertex_count);
        assert(count);
#endif
        return count;
      },
      ingpu_vertex_count, workload_offsets.GetArray(),
      workload_offsets.GetArray() + ingpu_vertex_count, context);

  size_t total_workload_count;
  DToH(&total_workload_count, workload_offsets.GetArray() + ingpu_vertex_count,
       1);
  GpuUtils::LoadBalance::LBSTransform(
      [=] DEVICE(int index, int seg, int rank) {
        uintV v = ingpu_vertex_ids_data[seg];
        size_t write_pos = new_row_ptrs_data[v] + rank;
        size_t read_pos = cur_row_ptrs[v] + rank;
        new_cols_data[write_pos] = cur_cols[read_pos];
      },
      total_workload_count, workload_offsets.GetArray(), ingpu_vertex_count,
      context);
}

// The subgraph represented by (toload_row_ptrs, load_cols),
// is the external subgraph loaded from main memory.
// Copy this subgraph into the new subgraph represented by
// (new_row_ptrs, new_cols)
static void LoadAdjListsFromMainMemory(
    size_t d_partition_id, CudaContext* context,
    DeviceArray<uintV>* toload_vertex_ids, DeviceArray<uintE>* toload_row_ptrs,
    DeviceArray<uintV>* load_cols, DeviceArray<uintE>* new_row_ptrs,
    DeviceArray<uintV>* new_cols, GPUProfiler* gpu_profiler,
    CountProfiler* count_profiler) {
  uintV* toload_vertex_ids_data = toload_vertex_ids->GetArray();
  uintE* toload_row_ptrs_data = toload_row_ptrs->GetArray();
  uintV* load_cols_data = load_cols->GetArray();
  uintE* new_row_ptrs_data = new_row_ptrs->GetArray();
  uintV* new_cols_data = new_cols->GetArray();

  GpuUtils::LoadBalance::LBSTransform(
      [=] DEVICE(int index, int seg, int rank) {
        uintV v1 = toload_vertex_ids_data[seg];
        uintV v2 = load_cols_data[index];
        new_cols_data[new_row_ptrs_data[v1] + rank] = v2;
      },
      load_cols->GetSize(), toload_row_ptrs->GetArray(),
      toload_vertex_ids->GetSize(), context);
}

// Load some adjacent lists from main memory (toload_vertex_ids),
// combine some adjacnet lists (ingpu_vertex_ids)
// from the current subgraph we have (graph_partition),
// to build the new subgraph we need.
static void IncBlockingBuildSubgraph(
    size_t d_partition_id, CudaContext* context,
    DevGraphPartition* graph_partition, DeviceArray<uintV>* required_vertex_ids,
    DeviceArray<uintV>* toload_vertex_ids, DeviceArray<uintV>* ingpu_vertex_ids,
    GraphDevTracker* graph_dev_tracker, TrackPartitionedGraph* cpu_relation,
    GPUProfiler* gpu_profiler, CountProfiler* count_profiler,
    size_t load_graph_thread_num) {
  gpu_profiler->StartTimer("prepare_load_graph_task_time", d_partition_id,
                           context->Stream());
  // required_row_ptrs is used by CPUs to make reading the
  // adjacent lists easier
  DeviceArray<uintE>* toload_row_ptrs = NULL;
  graph_dev_tracker->BuildSubgraphRowPtrs(toload_vertex_ids, toload_row_ptrs,
                                          context);

  gpu_profiler->StartTimer("prepare_load_graph_task_copy_allocate_time",
                           d_partition_id, context->Stream());
  // WARNING: allocate memory ad-hoc could be slow.
  // Consider use an efficient memory allocator
  uintV* h_toload_vertex_ids = new uintV[toload_vertex_ids->GetSize()];
  uintE* h_toload_row_ptrs = new uintE[toload_row_ptrs->GetSize()];
  DToH(h_toload_vertex_ids, toload_vertex_ids->GetArray(),
       toload_vertex_ids->GetSize());
  DToH(h_toload_row_ptrs, toload_row_ptrs->GetArray(),
       toload_row_ptrs->GetSize());
  gpu_profiler->EndTimer("prepare_load_graph_task_copy_allocate_time",
                         d_partition_id, context->Stream());

  gpu_profiler->EndTimer("prepare_load_graph_task_time", d_partition_id,
                         context->Stream());
  count_profiler->AddCount("load_subgraph_pcie_send", d_partition_id,
                           sizeof(uintV) * toload_vertex_ids->GetSize() +
                               sizeof(uintE) * toload_row_ptrs->GetSize());

  // submit the task to CPUs so that they can read the adjacent lists
  size_t load_vertex_count = toload_vertex_ids->GetSize();
  size_t load_edge_count = h_toload_row_ptrs[load_vertex_count];
  auto load_graph_job =
      std::async(std::launch::async, LoadSubgraph, cpu_relation,
                 load_vertex_count, load_edge_count, h_toload_vertex_ids,
                 h_toload_row_ptrs, load_graph_thread_num);

  // while waiting for the load subgraph task, asynchronously start
  // building subgraph and make prepartion

  // build row_ptrs for the new graph
  DeviceArray<uintE>* new_row_ptrs = NULL;
  BuildSubgraphRowPtrs(d_partition_id, context, required_vertex_ids,
                       graph_dev_tracker, new_row_ptrs, gpu_profiler,
                       count_profiler);

  // From the current subgraph in device memory, load the adj lists needed.
  // The remaining adj lists needed are loaded later from load_graph_task.
  uintE new_total_edge_count;
  DToH(&new_total_edge_count,
       new_row_ptrs->GetArray() + graph_dev_tracker->GetVertexCount(), 1);
  DeviceArray<uintV>* new_cols =
      new DeviceArray<uintV>(new_total_edge_count, context);
  LoadAdjListsInGPU(d_partition_id, context, graph_partition, ingpu_vertex_ids,
                    new_row_ptrs, new_cols, gpu_profiler, count_profiler);

  // the old subgraph is no more used
  graph_partition->Release();

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

  // Copy the graph data obtained from main memory to the
  // corresponding positions in our built subgraph
  LoadAdjListsFromMainMemory(d_partition_id, context, toload_vertex_ids,
                             toload_row_ptrs, load_cols, new_row_ptrs, new_cols,
                             gpu_profiler, count_profiler);

  // release
  delete toload_row_ptrs;
  toload_row_ptrs = NULL;
  delete load_cols;
  load_cols = NULL;

  delete[] h_toload_vertex_ids;
  h_toload_vertex_ids = NULL;
  delete[] h_toload_row_ptrs;
  h_toload_row_ptrs = NULL;
  delete[] load_graph_cols;
  load_graph_cols = NULL;

  graph_partition->Set(cpu_relation->GetVertexCount(), new_cols->GetSize(),
                       new_row_ptrs, new_cols);
}

// the entry point of incremental loading subgraph.
// We only load the necessary part of subgraph from main memory to reduce
// communication overhead.
template <typename BackwardNeighborGatherFunctor>
static void IncBlockingLoadSubgraph(
    size_t d_partition_id, CudaContext* context,
    DevGraphPartition* graph_partition, GraphDevTracker* graph_dev_tracker,
    TrackPartitionedGraph* cpu_relation,
    BackwardNeighborGatherFunctor backward_neighbor_functor, size_t path_num,
    GPUProfiler* gpu_profiler, CountProfiler* count_profiler,
    size_t load_graph_thread_num) {
  DeviceArray<uintV>* required_vertex_ids = NULL;
  DeviceArray<uintV>* toload_vertex_ids = NULL;
  DeviceArray<uintV>* ingpu_vertex_ids = NULL;

  gpu_profiler->StartTimer("inspect_join_time", d_partition_id,
                           context->Stream());
  IncInspectRequiredAdjLists(
      d_partition_id, context, graph_partition, graph_dev_tracker,
      required_vertex_ids, toload_vertex_ids, ingpu_vertex_ids,
      backward_neighbor_functor, path_num, gpu_profiler, count_profiler);
  gpu_profiler->EndTimer("inspect_join_time", d_partition_id,
                         context->Stream());

  assert(required_vertex_ids->GetSize() ==
         toload_vertex_ids->GetSize() + ingpu_vertex_ids->GetSize());

  if (toload_vertex_ids->GetSize() > 0) {
    gpu_profiler->StartTimer("build_subgraph_time", d_partition_id,
                             context->Stream());

    IncBlockingBuildSubgraph(
        d_partition_id, context, graph_partition, required_vertex_ids,
        toload_vertex_ids, ingpu_vertex_ids, graph_dev_tracker, cpu_relation,
        gpu_profiler, count_profiler, load_graph_thread_num);
    gpu_profiler->EndTimer("build_subgraph_time", d_partition_id,
                           context->Stream());
  }

  delete required_vertex_ids;
  required_vertex_ids = NULL;
  delete toload_vertex_ids;
  toload_vertex_ids = NULL;
  delete ingpu_vertex_ids;
  ingpu_vertex_ids = NULL;
}

#endif