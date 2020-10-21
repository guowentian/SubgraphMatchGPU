#ifndef __HYBRID_GPSM_PIPELINE_GPU_COMMON_CUH__
#define __HYBRID_GPSM_PIPELINE_GPU_COMMON_CUH__

#include "BatchData.h"
#include "BatchManager.cuh"
#include "CountProfiler.h"
#include "CudaContext.cuh"
#include "DevGraphPartition.cuh"
#include "DevTraversalPlan.cuh"
#include "DeviceArray.cuh"
#include "GPUProfiler.cuh"
#include "GraphDevTracker.cuh"
#include "TraversalPlan.h"
#include "WorkContext.cuh"

namespace Gpsm {
class ImResult {
 public:
  // Mainly used for pipelining to hold the current intermediate result
  ImResult(LayeredDeviceArray<uintV>& from, size_t count) {
    SwapInstances(from, count);
  }
  // initialize ImResult from scratch
  ImResult(TraversalPlan* plan) {
    size_t dimension = plan->GetVertexCount();
    d_inst_.resize(dimension, NULL);
  }
  ~ImResult() { Release(); }

  // copy the partial instances with the offsets specifed in the current batch
  void CopyBatchData(size_t dimension, ImResult* im_result,
                     BatchSpec* batch_spec) {
    auto& d_inst = im_result->GetInst();
    size_t batch_parent_count = batch_spec->GetBatchCount();
    size_t left_end = batch_spec->GetBatchLeftEnd();
    for (size_t l = 0; l < dimension; ++l) {
      ReAllocate(d_inst_[l], d_inst[l]->GetArray() + left_end,
                 batch_parent_count);
    }
  }

  void Release() {
    size_t dimension = d_inst_.size();
    for (size_t i = 0; i < dimension; ++i) {
      ReleaseIfExists(d_inst_[i]);
    }
  }

  void SwapInstances(LayeredDeviceArray<uintV>& from, size_t count) {
    d_inst_.resize(count, NULL);
    for (size_t i = 0; i < count; ++i) {
      std::swap(from[i], d_inst_[i]);
    }
  }

  LayeredDeviceArray<uintV>& GetInst() { return d_inst_; }

 protected:
  LayeredDeviceArray<uintV> d_inst_;
};

typedef ImResult ImData;

class ImDataDevHolder {
 public:
  ImDataDevHolder(size_t levels_num, CudaContext* context) {
    d_seq_inst_ = new DeviceArray<uintV*>(levels_num, context);
  }
  ~ImDataDevHolder() { ReleaseIfExists(d_seq_inst_); }

  void GatherImData(ImData* im_data, CudaContext* context) {
    AsyncCopyTwoDimensionDeviceArray(d_seq_inst_, im_data->GetInst(),
                                     context->Stream());
  }

  // getter
  DeviceArray<uintV*>* GetSeqInst() { return d_seq_inst_; }

 protected:
  DeviceArray<uintV*>* d_seq_inst_;
};

struct GpsmWorkContext : WorkContext {
  TraversalPlan* plan;
  ImData* im_data;
  ImDataDevHolder* im_data_holder;
  DevTraversalPlan* dev_plan;

  GpsmWorkContext() {
    plan = NULL;
    im_data = NULL;
    im_data_holder = NULL;
    dev_plan = NULL;
  }

  void Set(size_t d_partition_id, size_t thread_num,
           TrackPartitionedGraph* cpu_relation, long long* ans,
           CudaContext* context, DevGraphPartition* graph_partition,
           GPUProfiler* gpu_profiler, CountProfiler* count_profiler,
           TraversalPlan* plan, ImData* im_data,
           ImDataDevHolder* im_data_holder, DevTraversalPlan* dev_plan) {
    WorkContext::Set(d_partition_id, thread_num, cpu_relation, ans, context,
                     graph_partition, gpu_profiler, count_profiler);
    this->plan = plan;
    this->im_data = im_data;
    this->im_data_holder = im_data_holder;
    this->dev_plan = dev_plan;
  }
};

struct ImBatchData : BatchData {
  ImBatchData() { im_result = NULL; }
  ImResult* im_result;
};

}  // namespace Gpsm

#endif
