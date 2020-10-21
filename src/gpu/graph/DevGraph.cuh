#ifndef __GPU_DEV_GRAPH_CUH__
#define __GPU_DEV_GRAPH_CUH__

#include "AbstractGraph.h"
#include "CPUGraph.h"
#include "CudaContext.cuh"
#include "DeviceArray.cuh"
#include "GPUUtil.cuh"
#include "Meta.h"

class DevGraph : public AbstractGraph {
 public:
  DevGraph(Graph* cpu_graph, CudaContext* context)
      : AbstractGraph(cpu_graph->GetDirected()), context_(context) {
    BuildDevGraph(cpu_graph, context_);
  }
  ~DevGraph() {
    delete d_row_ptrs_;
    d_row_ptrs_ = NULL;
    delete d_cols_;
    d_cols_ = NULL;
  }

  DeviceArray<uintE>* GetRowPtrs() const { return d_row_ptrs_; }
  DeviceArray<uintV>* GetCols() const { return d_cols_; }

 protected:
  CudaContext* context_;
  DeviceArray<uintE>* d_row_ptrs_;
  DeviceArray<uintV>* d_cols_;

  void BuildDevGraph(Graph* cpu_graph, CudaContext* context) {
    cudaStream_t stream = context->Stream();
    vertex_count_ = cpu_graph->GetVertexCount();
    edge_count_ = cpu_graph->GetEdgeCount();
    d_row_ptrs_ = new DeviceArray<uintE>(vertex_count_ + 1, context);
    d_cols_ = new DeviceArray<uintV>(edge_count_, context);

    CUDA_ERROR(cudaMemcpyAsync(d_row_ptrs_->GetArray(), cpu_graph->GetRowPtrs(),
                               sizeof(uintE) * (vertex_count_ + 1),
                               cudaMemcpyHostToDevice, stream));
    CUDA_ERROR(cudaMemcpyAsync(d_cols_->GetArray(), cpu_graph->GetCols(),
                               sizeof(uintV) * edge_count_,
                               cudaMemcpyHostToDevice, stream));
  }
};

class TrackPartitionedDevGraph : public DevGraph {
 public:
  TrackPartitionedDevGraph(TrackPartitionedGraph* cpu_graph,
                           CudaContext* context)
      : DevGraph(cpu_graph, context) {
    BuildTrackPartitionedDevGraph(cpu_graph, context);
  }
  ~TrackPartitionedDevGraph() {
    delete d_vertex_partition_map_;
    d_vertex_partition_map_ = NULL;
    delete d_inter_row_ptrs_;
    d_inter_row_ptrs_ = NULL;
    delete d_inter_cols_;
    d_inter_cols_ = NULL;
  }

  size_t GetInterPartitionEdgesCount() const { return inter_edges_count_; }
  DeviceArray<uintE>* GetInterRowPtrs() const { return d_inter_row_ptrs_; }
  DeviceArray<uintV>* GetInterCols() const { return d_inter_cols_; }
  DeviceArray<uintP>* GetVertexPartitionMap() const {
    return d_vertex_partition_map_;
  }

 private:
  size_t inter_edges_count_;
  DeviceArray<uintP>* d_vertex_partition_map_;
  DeviceArray<uintE>* d_inter_row_ptrs_;
  DeviceArray<uintV>* d_inter_cols_;

  void BuildTrackPartitionedDevGraph(TrackPartitionedGraph* cpu_graph,
                                     CudaContext* context) {
    cudaStream_t stream = context->Stream();
    inter_edges_count_ = cpu_graph->GetInterPartitionEdgesCount();
    d_vertex_partition_map_ =
        new DeviceArray<uintP>(cpu_graph->GetVertexCount(), context);
    d_inter_row_ptrs_ =
        new DeviceArray<uintE>(cpu_graph->GetVertexCount() + 1, context);
    d_inter_cols_ = new DeviceArray<uintV>(
        cpu_graph->GetInterPartitionEdgesCount(), context);
    CUDA_ERROR(cudaMemcpyAsync(d_vertex_partition_map_->GetArray(),
                               cpu_graph->GetVertexPartitionMap(),
                               sizeof(uintP) * cpu_graph->GetVertexCount(),
                               cudaMemcpyHostToDevice, stream));
    CUDA_ERROR(cudaMemcpyAsync(
        d_inter_row_ptrs_->GetArray(), cpu_graph->GetInterRowPtrs(),
        sizeof(uintE) * (cpu_graph->GetVertexCount() + 1),
        cudaMemcpyHostToDevice, stream));
    CUDA_ERROR(cudaMemcpyAsync(
        d_inter_cols_->GetArray(), cpu_graph->GetInterCols(),
        sizeof(uintV) * cpu_graph->GetInterPartitionEdgesCount(),
        cudaMemcpyHostToDevice, stream));
  }
};

#endif
