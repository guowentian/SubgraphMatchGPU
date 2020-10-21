#ifndef __GPU_GRAPH_DEVICE_TRACKER_CUH__
#define __GPU_GRAPH_DEVICE_TRACKER_CUH__

#include "DeviceArray.cuh"
#include "Scan.cuh"

// GraphDevTracker stores some data structures about the global graph stored in
// the host. Due to the limited device memory, GPU only stores a subgraph and
// needs to fetch graph data from the host. To do this, we need some information
// about the global graph so that GPUs know how much data can be fetched and
// make preparation.
class GraphDevTracker {
 public:
  GraphDevTracker(uintP* partition_ids, uintE* row_ptrs, size_t vertex_count,
                  size_t edge_count, CudaContext* context)
      : vertex_count_(vertex_count), edge_count_(edge_count) {
#if defined(UNTRACK_ALLOC_LARGE_VERTEX_NUM)
    /// A temporary fix: to handle the large graphs that have a lot of vertices,
    /// avoid tracking memory statistics when allocating d_partition_ids_ and
    /// d_graph_row_ptrs_. The two arrays need to be maintained in GPU memory
    /// during inter-partition search.

    // TODO: a systematic way is to
    // store d_graph_row_ptrs_ in main memory and load it on demand
    void* mem1 = context->UnTrackMalloc(vertex_count * sizeof(uintP));
    d_partition_ids_ =
        new DeviceArray<uintP>((uintP*)mem1, vertex_count, context, false);
    void* mem2 = context->UnTrackMalloc((vertex_count + 1) * sizeof(uintE));
    d_graph_row_ptrs_ =
        new DeviceArray<uintE>((uintE*)mem2, vertex_count + 1, context, false);
#else
    d_partition_ids_ = new DeviceArray<uintP>(vertex_count, context);
    d_graph_row_ptrs_ = new DeviceArray<uintE>(vertex_count + 1, context);
#endif
    HToD(d_partition_ids_->GetArray(), partition_ids, vertex_count);
    HToD(d_graph_row_ptrs_->GetArray(), row_ptrs, vertex_count + 1);
  }
  ~GraphDevTracker() {
#if defined(UNTRACK_ALLOC_LARGE_VERTEX_NUM)
    void* mem1 = d_partition_ids_->GetArray();
    d_partition_ids_->GetContext()->UnTrackFree(mem1);
    void* mem2 = d_graph_row_ptrs_->GetArray();
    d_graph_row_ptrs_->GetContext()->UnTrackFree(mem2);
#endif
    delete d_partition_ids_;
    d_partition_ids_ = NULL;
    delete d_graph_row_ptrs_;
    d_graph_row_ptrs_ = NULL;
  }

  void BuildSubgraphRowPtrs(DeviceArray<uintV>* d_vertex_ids,
                            DeviceArray<uintE>*& row_ptrs,
                            CudaContext* context) {
    uintV* d_vertex_ids_data = d_vertex_ids->GetArray();
    size_t count = d_vertex_ids->GetSize();
    uintE* d_graph_row_ptrs_data = d_graph_row_ptrs_->GetArray();
    row_ptrs = new DeviceArray<uintE>(count + 1, context);
    GpuUtils::Scan::TransformScan(
        [=] DEVICE(int index) {
          uintV v = d_vertex_ids_data[index];
          uintE c = d_graph_row_ptrs_data[v + 1] - d_graph_row_ptrs_data[v];
          return c;
        },
        count, row_ptrs->GetArray(), row_ptrs->GetArray() + count, context);
  }

  DeviceArray<uintP>* GetPartitionIds() const { return d_partition_ids_; }
  DeviceArray<uintE>* GetGraphRowPtrs() const { return d_graph_row_ptrs_; }
  size_t GetVertexCount() const { return vertex_count_; }
  size_t GetEdgeCount() const { return edge_count_; }

 private:
  size_t vertex_count_;
  size_t edge_count_;
  // of size |V|
  DeviceArray<uintP>* d_partition_ids_;
  // adjacent list size of global graph
  DeviceArray<uintE>* d_graph_row_ptrs_;
};

#endif
