#ifndef __HYBRID_GPSM_REUSE_GPU_COMMON_CUH__
#define __HYBRID_GPSM_REUSE_GPU_COMMON_CUH__

#include "CudaContext.cuh"
#include "GPSMCommon.cuh"

namespace GpsmReuse {
using Gpsm::GpsmWorkContext;
using Gpsm::ImData;
using Gpsm::ImDataDevHolder;
using Gpsm::ImResult;

class CacheData {
 public:
  CacheData(size_t levels_num, CudaContext* context) : levels_num_(levels_num) {
    d_inst_next_offsets_.resize(levels_num_, NULL);
    d_inst_parents_indices_.resize(levels_num_, NULL);
    d_inst_ptrs_.resize(levels_num_, NULL);
    d_cache_next_offsets_.resize(levels_num_, NULL);
    d_cache_instances_.resize(levels_num_, NULL);
    level_batch_offsets_start_.resize(levels_num_, 0);
    level_batch_offsets_end_.resize(levels_num_, 0);

    d_seq_inst_next_offsets_ = new DeviceArray<size_t*>(levels_num_, context);
    d_seq_inst_parents_indices_ =
        new DeviceArray<size_t*>(levels_num_, context);
    d_seq_cache_next_offsets_ = new DeviceArray<size_t*>(levels_num_, context);
    d_seq_cache_instances_ = new DeviceArray<uintV*>(levels_num_, context);
    d_seq_inst_ptrs_ = new DeviceArray<uintV*>(levels_num_, context);
    d_seq_level_batch_offsets_start_ =
        new DeviceArray<size_t>(levels_num_, context);
    d_seq_level_batch_offsets_end_ =
        new DeviceArray<size_t>(levels_num_, context);
  }
  ~CacheData() {
    d_inst_ptrs_.clear();
    for (size_t i = 0; i < levels_num_; ++i) {
      ReleaseIfExists(d_inst_next_offsets_[i]);
      ReleaseIfExists(d_inst_parents_indices_[i]);
      ReleaseIfExists(d_cache_next_offsets_[i]);
      ReleaseIfExists(d_cache_instances_[i]);
    }

    delete d_seq_inst_next_offsets_;
    d_seq_inst_next_offsets_ = NULL;
    delete d_seq_inst_parents_indices_;
    d_seq_inst_parents_indices_ = NULL;
    delete d_seq_cache_next_offsets_;
    d_seq_cache_next_offsets_ = NULL;
    delete d_seq_cache_instances_;
    d_seq_cache_instances_ = NULL;
    delete d_seq_inst_ptrs_;
    d_seq_inst_ptrs_ = NULL;
    delete d_seq_level_batch_offsets_start_;
    d_seq_level_batch_offsets_start_ = NULL;
    delete d_seq_level_batch_offsets_end_;
    d_seq_level_batch_offsets_end_ = NULL;
  }

  void GatherCacheData(CudaContext* context) {
    cudaStream_t stream = context->Stream();
    AsyncCopyTwoDimensionDeviceArray(d_seq_inst_next_offsets_,
                                     d_inst_next_offsets_, stream);
    AsyncCopyTwoDimensionDeviceArray(d_seq_inst_parents_indices_,
                                     d_inst_parents_indices_, stream);
    AsyncCopyTwoDimensionDeviceArray(d_seq_cache_next_offsets_,
                                     d_cache_next_offsets_, stream);
    AsyncCopyTwoDimensionDeviceArray(d_seq_cache_instances_, d_cache_instances_,
                                     stream);
    HToD(d_seq_inst_ptrs_->GetArray(), d_inst_ptrs_.data(), levels_num_,
         stream);
    HToD(d_seq_level_batch_offsets_start_->GetArray(),
         level_batch_offsets_start_.data(), levels_num_, stream);
    HToD(d_seq_level_batch_offsets_end_->GetArray(),
         level_batch_offsets_end_.data(), levels_num_, stream);
  }

  // Setter function
  void SetLevelBatchoffset(size_t level, size_t start, size_t end) {
    level_batch_offsets_start_[level] = start;
    level_batch_offsets_end_[level] = end;
  }

  // getter
  LayeredDeviceArray<size_t>& GetInstNextOffsets() {
    return d_inst_next_offsets_;
  }
  LayeredDeviceArray<size_t>& GetInstParentsIndices() {
    return d_inst_parents_indices_;
  }
  LayeredDeviceArray<size_t>& GetCacheNextOffsets() {
    return d_cache_next_offsets_;
  }
  LayeredDeviceArray<uintV>& GetCacheInstances() { return d_cache_instances_; }
  std::vector<uintV*>& GetInstPtrs() { return d_inst_ptrs_; }

  DeviceArray<size_t*>* GetSeqInstNextOffsets() const {
    return d_seq_inst_next_offsets_;
  }
  DeviceArray<size_t*>* GetSeqInstParentsIndices() const {
    return d_seq_inst_parents_indices_;
  }
  DeviceArray<size_t*>* GetSeqCacheNextOffsets() const {
    return d_seq_cache_next_offsets_;
  }
  DeviceArray<uintV*>* GetSeqCacheInstances() const {
    return d_seq_cache_instances_;
  }
  DeviceArray<size_t>* GetSeqLevelBatchOffsetsStart() const {
    return d_seq_level_batch_offsets_start_;
  }
  DeviceArray<size_t>* GetSeqLevelBatchOffsetsEnd() const {
    return d_seq_level_batch_offsets_end_;
  }
  DeviceArray<uintV*>* GetSeqInstPtrs() const { return d_seq_inst_ptrs_; }

 protected:
  size_t levels_num_;
  // to maintain the tree structure to retrieve the past result
  LayeredDeviceArray<size_t> d_inst_next_offsets_;
  LayeredDeviceArray<size_t> d_inst_parents_indices_;
  // d_inst_ptrs_[i] is the array storing all unique partial instance in the
  // i-th level, and this array only stores the data vertices matched to the
  // i-th level.
  std::vector<uintV*> d_inst_ptrs_;

  // CSR-like structure to keep intersection result
  LayeredDeviceArray<size_t> d_cache_next_offsets_;
  LayeredDeviceArray<uintV> d_cache_instances_;

  // Due to batching, only some partial instances at a level would be expanded.
  // Therefore, given a index at a level, need to
  // use pos=(index-level_batch_offsets_[level]) to access d_next_offsets_.
  // Inclusive interval: [level_batch_offsets_start_, level_batch_offsets_end_]
  std::vector<size_t> level_batch_offsets_start_;
  std::vector<size_t> level_batch_offsets_end_;

  DeviceArray<size_t*>* d_seq_inst_next_offsets_;
  DeviceArray<size_t*>* d_seq_inst_parents_indices_;
  DeviceArray<size_t*>* d_seq_cache_next_offsets_;
  DeviceArray<uintV*>* d_seq_cache_instances_;
  DeviceArray<size_t>* d_seq_level_batch_offsets_start_;
  DeviceArray<size_t>* d_seq_level_batch_offsets_end_;
  DeviceArray<uintV*>* d_seq_inst_ptrs_;
};

template <typename CacheDataT>
struct GpsmReuseWorkContextGeneral : public GpsmWorkContext {
  CacheDataT* cache_data;

  GpsmReuseWorkContextGeneral() : GpsmWorkContext() { cache_data = NULL; }

  void Set(size_t d_partition_id, size_t thread_num,
           TrackPartitionedGraph* cpu_relation, long long* ans,
           CudaContext* context, DevGraphPartition* graph_partition,
           GPUProfiler* gpu_profiler, CountProfiler* count_profiler,
           TraversalPlan* plan, ImData* im_data,
           ImDataDevHolder* im_data_holder, DevTraversalPlan* dev_plan,
           CacheDataT* cache_data) {
    GpsmWorkContext::Set(d_partition_id, thread_num, cpu_relation, ans, context,
                         graph_partition, gpu_profiler, count_profiler, plan,
                         im_data, im_data_holder, dev_plan);
    this->cache_data = cache_data;
  }
};

typedef GpsmReuseWorkContextGeneral<CacheData> GpsmReuseWorkContext;

}  // namespace GpsmReuse

#endif
