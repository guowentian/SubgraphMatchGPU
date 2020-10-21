#ifndef __GPU_DEV_GRAPH_PARTITION_CUH__
#define __GPU_DEV_GRAPH_PARTITION_CUH__

#include "CPUGraph.h"
#include "DeviceArray.cuh"
#include "GPUUtil.cuh"
#include "GraphPartition.h"
#include "Meta.h"

class DevGraphPartition {
 public:
  DevGraphPartition(CudaContext *context) : context_(context) {
    vertex_count_ = edge_count_ = 0;
    d_row_ptrs_ = NULL;
    d_cols_ = NULL;
  }
  ~DevGraphPartition() { Release(); }
  void ReAllocate(GraphPartition *cpu_partition) {
    if (d_row_ptrs_ == NULL ||
        cpu_partition->GetVertexCount() + 1 > d_row_ptrs_->GetSize()) {
      ::ReAllocate(d_row_ptrs_, cpu_partition->GetVertexCount() + 1, context_);
    }
    if (d_cols_ == NULL || cpu_partition->GetEdgeCount() > d_cols_->GetSize()) {
      ::ReAllocate(d_cols_, cpu_partition->GetEdgeCount(), context_);
    }
  }
  void Release() {
    vertex_count_ = edge_count_ = 0;
    ReleaseIfExists(d_row_ptrs_);
    ReleaseIfExists(d_cols_);
  }
  void BuildIntraPartition(GraphPartition *cpu_partition) {
    vertex_count_ = cpu_partition->GetVertexCount();
    edge_count_ = cpu_partition->GetEdgeCount();
    this->ReAllocate(cpu_partition);

    HToD(d_row_ptrs_->GetArray(), cpu_partition->GetRowPtrs(),
         vertex_count_ + 1, context_->Stream());
    HToD(d_cols_->GetArray(), cpu_partition->GetCols(), edge_count_,
         context_->Stream());
  }

  void UnTrackCopyHToD(Graph *h_graph) {
    size_t vertex_count = h_graph->GetVertexCount();
    size_t edge_count = h_graph->GetEdgeCount();
    uintE *d_row_ptrs_mem =
        (uintE *)context_->UnTrackMalloc(sizeof(uintE) * (vertex_count + 1));
    uintV *d_cols_mem =
        (uintV *)context_->UnTrackMalloc(sizeof(uintV) * edge_count);
    DeviceArray<uintE> *d_row_ptrs = new DeviceArray<uintE>(
        d_row_ptrs_mem, vertex_count + 1, context_, false);
    DeviceArray<uintV> *d_cols =
        new DeviceArray<uintV>(d_cols_mem, edge_count, context_, false);
    HToD(d_row_ptrs->GetArray(), h_graph->GetRowPtrs(), vertex_count + 1);
    HToD(d_cols->GetArray(), h_graph->GetCols(), edge_count);
    this->Set(vertex_count, edge_count, d_row_ptrs, d_cols);
  }

  CudaContext *GetContext() const { return context_; }
  size_t GetVertexCount() const { return vertex_count_; }
  size_t GetEdgeCount() const { return edge_count_; }

  DeviceArray<uintE> *GetRowPtrs() const { return d_row_ptrs_; }
  DeviceArray<uintV> *GetCols() const { return d_cols_; }

  void Set(size_t vertex_count, size_t edge_count,
           DeviceArray<uintE> *d_row_ptrs, DeviceArray<uintV> *d_cols) {
    assert(d_row_ptrs_ == NULL);
    assert(d_cols_ == NULL);
    assert(d_row_ptrs->GetSize() == vertex_count + 1);
    assert(d_cols->GetSize() == edge_count);

    vertex_count_ = vertex_count;
    edge_count_ = edge_count;
    d_row_ptrs_ = d_row_ptrs;
    d_cols_ = d_cols;
  }
  bool Empty() const { return d_row_ptrs_ == NULL; }

 private:
  CudaContext *context_;
  size_t vertex_count_;
  size_t edge_count_;

  DeviceArray<uintE> *d_row_ptrs_;
  DeviceArray<uintV> *d_cols_;
};

#endif
