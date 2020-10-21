#ifndef __HYBRID_GPSM_REUSE_GPU_COMPONENT_CUH__
#define __HYBRID_GPSM_REUSE_GPU_COMPONENT_CUH__

#include "HybridGPSMGPUComp.cuh"
#include "HybridGPSMReuseGPUProcessor.cuh"

namespace GpsmReuse {
class HybridGpsmReusePipelineGPUComponent
    : public Gpsm::HybridGpsmPipelineGPUComponent {
 public:
  HybridGpsmReusePipelineGPUComponent(ReuseTraversalPlan* plan,
                                      TrackPartitionedGraph* rel,
                                      bool materialize_result,
                                      size_t thread_num)
      : Gpsm::HybridGpsmPipelineGPUComponent(plan, rel, materialize_result,
                                             thread_num),
        plan_(plan) {
    ProfilerHelper::AddGPUProfilePhaseReuse(gpu_profiler_);
    gpu_profiler_->AddPhase("join_phase_prepare_time");

    ProfilerHelper::AddCountProfilePhaseReuse(count_profiler_);
  }
  ~HybridGpsmReusePipelineGPUComponent() {
    size_t partition_num = plan_->GetDevPartitionNum();
    for (size_t dev_id = 0; dev_id < partition_num; ++dev_id) {
      CUDA_ERROR(cudaSetDevice(dev_id));
      delete dev_plans_[dev_id];
      dev_plans_[dev_id] = NULL;
      delete cache_data_[dev_id];
      cache_data_[dev_id] = NULL;
    }
  }

  virtual WorkContext* CreateWorkContext(Task* task) {
    size_t d_partition_id = task->d_partition_id_;
    GpsmReuseWorkContext* wctx = new GpsmReuseWorkContext();
    wctx->Set(d_partition_id, thread_num_, cpu_relation_, task->ans_,
              cuda_contexts_[d_partition_id], d_partitions_[d_partition_id],
              gpu_profiler_, count_profiler_, plan_, im_data_[d_partition_id],
              im_data_holder_[d_partition_id], dev_plans_[d_partition_id],
              cache_data_[d_partition_id]);

    return wctx;
  }

  virtual HybridGPUProcessor* CreateGPUProcessor(WorkContext* wctx,
                                                 Task* task) {
    GpsmReuseWorkContext* gpsm_wctx = static_cast<GpsmReuseWorkContext*>(wctx);
    return new HybridGPSMReuseGPUProcessor(gpsm_wctx, task);
  }

  virtual void InitGPU() {
    Gpsm::HybridGpsmPipelineGPUComponent::InitGPU();
    size_t partition_num = plan_->GetDevPartitionNum();
    dev_plans_.resize(partition_num, NULL);
    cache_data_.resize(partition_num, NULL);
    for (size_t dev_id = 0; dev_id < partition_num; ++dev_id) {
      CUDA_ERROR(cudaSetDevice(dev_id));
      dev_plans_[dev_id] =
          new DevReuseTraversalPlan(plan_, cuda_contexts_[dev_id]);
      cache_data_[dev_id] =
          new CacheData(plan_->GetVertexCount(), cuda_contexts_[dev_id]);
    }
  }

 private:
  ReuseTraversalPlan* plan_;
  std::vector<DevReuseTraversalPlan*> dev_plans_;
  std::vector<CacheData*> cache_data_;
  // inherit im_data_ and im_data_holder_ from HybridGPSMGPUComp
};
}  // namespace GpsmReuse

#endif
