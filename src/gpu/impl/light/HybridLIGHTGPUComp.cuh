#ifndef __HYBRID_LIGHT_PIPELINE_GPU_COMPONENT_CUH__
#define __HYBRID_LIGHT_PIPELINE_GPU_COMPONENT_CUH__

#include "HybridLIGHTGPUProcessor.cuh"
#include "LazyTraversalPlan.h"

namespace Light {
class HybridLIGHTPipelineGPUComponent : public HybridGPUComponent {
 public:
  HybridLIGHTPipelineGPUComponent(LazyTraversalPlan *plan,
                                  TrackPartitionedGraph *rel, bool materialize,
                                  size_t thread_num)
      : HybridGPUComponent(plan, rel, materialize, thread_num), plan_(plan) {
    ProfilerHelper::AddGPUProfilePhaseLight(gpu_profiler_);
    ProfilerHelper::AddCountProfilePhaseLight(count_profiler_);
  }

  ~HybridLIGHTPipelineGPUComponent() {
    size_t p = plan_->GetDevPartitionNum();
    for (size_t dev_id = 0; dev_id < p; ++dev_id) {
      CUDA_ERROR(cudaSetDevice(dev_id));
      delete dev_plan_[dev_id];
      dev_plan_[dev_id] = NULL;
      delete im_data_[dev_id];
      im_data_[dev_id] = NULL;
      delete im_data_holder_[dev_id];
      im_data_holder_[dev_id] = NULL;
    }
  }

  virtual WorkContext *CreateWorkContext(Task *task) {
    LightWorkContext *wctx = new LightWorkContext();
    size_t d_partition_id = task->d_partition_id_;
    wctx->Set(d_partition_id, thread_num_, cpu_relation_, task->ans_,
              cuda_contexts_[d_partition_id], d_partitions_[d_partition_id],
              gpu_profiler_, count_profiler_, plan_, im_data_[d_partition_id],
              im_data_holder_[d_partition_id], dev_plan_[d_partition_id], NULL);
    return wctx;
  }

  virtual HybridGPUProcessor *CreateGPUProcessor(WorkContext *wctx0,
                                                 Task *task) {
    LightWorkContext *wctx = static_cast<LightWorkContext *>(wctx0);
    return new HybridLIGHTGPUProcessor(wctx, task);
  }

  virtual void InitGPU() {
    size_t partition_num = plan_->GetDevPartitionNum();
    dev_plan_.resize(partition_num);
    im_data_.resize(partition_num);
    im_data_holder_.resize(partition_num);

    for (size_t dev_id = 0; dev_id < partition_num; ++dev_id) {
      CUDA_ERROR(cudaSetDevice(dev_id));
      dev_plan_[dev_id] =
          new DevLazyTraversalPlan(plan_, cuda_contexts_[dev_id]);
      im_data_[dev_id] = new ImData(plan_->GetVertexCount());
      im_data_holder_[dev_id] =
          new ImDataDevHolder(plan_->GetVertexCount(), cuda_contexts_[dev_id]);
    }
  }

  std::vector<ImData *> &GetImData() { return im_data_; }
  std::vector<ImDataDevHolder *> &GetImDataDevHolder() {
    return im_data_holder_;
  }
  std::vector<DevLazyTraversalPlan *> GetDevicePlan() { return dev_plan_; }

 private:
  LazyTraversalPlan *plan_;
  std::vector<DevLazyTraversalPlan *> dev_plan_;

 protected:
  std::vector<ImData *> im_data_;
  std::vector<ImDataDevHolder *> im_data_holder_;
};
}  // namespace Light

#endif
