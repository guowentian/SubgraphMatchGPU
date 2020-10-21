#ifndef __HYBRID_LIGHT_PIPELINE_GPU_COMPONENT_COMMON_CUH__
#define __HYBRID_LIGHT_PIPELINE_GPU_COMPONENT_COMMON_CUH__

#include <vector>

#include "BatchData.h"
#include "BatchManager.cuh"
#include "CudaContext.cuh"
#include "DeviceArray.cuh"
#include "WorkContext.cuh"

#include "CountProfiler.h"
#include "DevLazyTraversalPlan.cuh"
#include "GPUProfiler.cuh"
#include "GraphDevTracker.cuh"
#include "LazyTraversalPlan.h"

namespace Light {
class ImResult {
 public:
  ImResult(ImResult* from, const VTGroup& materialized_vertices,
           const VTGroup& computed_unmaterialized_vertices, size_t n) {
    this->Swap(from, materialized_vertices, computed_unmaterialized_vertices,
               n);
  }
  ImResult(size_t n) {
    d_instances_.resize(n, NULL);
    d_candidates_.resize(n, NULL);
    d_candidates_offsets_.resize(n, NULL);
    d_candidates_indices_.resize(n, NULL);
  }
  ~ImResult() { Release(); }

  void CopyBatchData(ImResult* from, BatchSpec* batch_spec,
                     const VTGroup& materialized_vertices,
                     const VTGroup& computed_unmaterialized_vertices) {
    auto& from_instances = from->GetInstances();
    auto& from_candidates = from->GetCandidates();
    auto& from_candidates_offsets = from->GetCandidatesOffsets();
    auto& from_candidates_indices = from->GetCandidatesIndices();

    size_t batch_parent_count = batch_spec->GetBatchCount();
    size_t left_end = batch_spec->GetBatchLeftEnd();

    for (auto u : materialized_vertices) {
      ReAllocate(d_instances_[u], from_instances[u]->GetArray() + left_end,
                 batch_parent_count);
    }

    for (auto u : computed_unmaterialized_vertices) {
      ReAllocate(d_candidates_indices_[u],
                 from_candidates_indices[u]->GetArray() + left_end,
                 batch_parent_count);

      // keep pointer to the whole array
      if (from_candidates[u]) {
        // it is possible from_candidates[u]=NULL for COMPUTE_PATH_COUNT
        ReAllocate(d_candidates_[u], from_candidates[u]->GetArray(),
                   from_candidates[u]->GetSize());
      }
      ReAllocate(d_candidates_offsets_[u],
                 from_candidates_offsets[u]->GetArray(),
                 from_candidates_offsets[u]->GetSize());
    }
  }

  void Release() {
    size_t n = d_instances_.size();
    for (size_t i = 0; i < n; ++i) {
      ReleaseIfExists(d_instances_[i]);
      ReleaseIfExists(d_candidates_[i]);
      ReleaseIfExists(d_candidates_offsets_[i]);
      ReleaseIfExists(d_candidates_indices_[i]);
    }
  }
  void Swap(ImResult* from, const VTGroup& materialized_vertices,
            const VTGroup& computed_unmaterialized_vertices, size_t n) {
    auto& from_instances = from->GetInstances();
    auto& from_candidates = from->GetCandidates();
    auto& from_candidates_offsets = from->GetCandidatesOffsets();
    auto& from_candidates_indices = from->GetCandidatesIndices();

    d_instances_.resize(n, NULL);
    d_candidates_.resize(n, NULL);
    d_candidates_offsets_.resize(n, NULL);
    d_candidates_indices_.resize(n, NULL);

    for (auto u : materialized_vertices) {
      std::swap(d_instances_[u], from_instances[u]);
    }

    for (auto u : computed_unmaterialized_vertices) {
      std::swap(d_candidates_[u], from_candidates[u]);
      std::swap(d_candidates_offsets_[u], from_candidates_offsets[u]);
      std::swap(d_candidates_indices_[u], from_candidates_indices[u]);
    }
  }

  // ===== getter =====
  LayeredDeviceArray<uintV>& GetInstances() { return d_instances_; }
  LayeredDeviceArray<uintV>& GetCandidates() { return d_candidates_; }
  LayeredDeviceArray<size_t>& GetCandidatesOffsets() {
    return d_candidates_offsets_;
  }
  LayeredDeviceArray<size_t>& GetCandidatesIndices() {
    return d_candidates_indices_;
  }

 protected:
  LayeredDeviceArray<uintV> d_instances_;
  // candidate set
  LayeredDeviceArray<uintV> d_candidates_;
  LayeredDeviceArray<size_t> d_candidates_offsets_;
  LayeredDeviceArray<size_t> d_candidates_indices_;
};

typedef ImResult ImData;

class ImDataDevHolder {
 public:
  ImDataDevHolder(size_t n, CudaContext* context) {
    d_seq_instances_ = new DeviceArray<uintV*>(n, context);
    d_seq_candidates_ = new DeviceArray<uintV*>(n, context);
    d_seq_candidates_offsets_ = new DeviceArray<size_t*>(n, context);
    d_seq_candidates_indices_ = new DeviceArray<size_t*>(n, context);
  }
  ~ImDataDevHolder() {
    ReleaseIfExists(d_seq_instances_);
    ReleaseIfExists(d_seq_candidates_);
    ReleaseIfExists(d_seq_candidates_offsets_);
    ReleaseIfExists(d_seq_candidates_indices_);
  }

  void GatherImData(ImData* im_data, CudaContext* context) {
    BuildTwoDimensionDeviceArray(d_seq_instances_, &im_data->GetInstances(),
                                 context);
    BuildTwoDimensionDeviceArray(d_seq_candidates_, &im_data->GetCandidates(),
                                 context);
    BuildTwoDimensionDeviceArray(d_seq_candidates_offsets_,
                                 &im_data->GetCandidatesOffsets(), context);
    BuildTwoDimensionDeviceArray(d_seq_candidates_indices_,
                                 &im_data->GetCandidatesIndices(), context);
  }

  // ========= getter ===== ===
  DeviceArray<uintV*>* GetSeqInstances() const { return d_seq_instances_; }
  DeviceArray<uintV*>* GetSeqCandidates() const { return d_seq_candidates_; }
  DeviceArray<size_t*>* GetSeqCandidatesOffsets() const {
    return d_seq_candidates_offsets_;
  }
  DeviceArray<size_t*>* GetSeqCandidatesIndices() const {
    return d_seq_candidates_indices_;
  }

 protected:
  DeviceArray<uintV*>* d_seq_instances_;
  DeviceArray<uintV*>* d_seq_candidates_;
  DeviceArray<size_t*>* d_seq_candidates_offsets_;
  DeviceArray<size_t*>* d_seq_candidates_indices_;
};

struct LightWorkContext : WorkContext {
  LazyTraversalPlan* plan;
  ImData* im_data;
  ImDataDevHolder* im_data_holder;

  DevLazyTraversalPlan* dev_plan;
  GraphDevTracker* graph_dev_tracker;

  LightWorkContext() : WorkContext() {
    plan = NULL;
    im_data = NULL;
    im_data_holder = NULL;
    dev_plan = NULL;
    graph_dev_tracker = NULL;
  }

  void Set(size_t _d_partition_id, size_t _thread_num,
           TrackPartitionedGraph* _cpu_relation, long long* _ans,
           CudaContext* _context, DevGraphPartition* _graph_partition,
           GPUProfiler* _gpu_profiler, CountProfiler* _count_profiler,
           LazyTraversalPlan* _plan, ImData* _im_data,
           ImDataDevHolder* _im_data_holder, DevLazyTraversalPlan* _dev_plan,
           GraphDevTracker* _graph_dev_tracker) {
    WorkContext::Set(_d_partition_id, _thread_num, _cpu_relation, _ans,
                     _context, _graph_partition, _gpu_profiler,
                     _count_profiler);

    plan = _plan;
    im_data = _im_data;
    im_data_holder = _im_data_holder;
    dev_plan = _dev_plan;
    graph_dev_tracker = _graph_dev_tracker;
  }
};

struct ImBatchData : BatchData {
  ImBatchData() : im_result(NULL) {}
  ImResult* im_result;
};

}  // namespace Light

#endif
