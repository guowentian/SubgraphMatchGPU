#ifndef __HYBRID_GPSM_GPU_COMPONENT_CUH__
#define __HYBRID_GPSM_GPU_COMPONENT_CUH__

#include "HybridGPSMGPUProcessor.cuh"
#include "HybridGPUComp.cuh"

// Faithfully implement the paper
// "Fast subgraph matching on large graphs using graphics processors" DASFAA
// 2015. The salient feature is the warp-based processing and two-step output
// scheme.

namespace Gpsm {

class HybridGpsmPipelineGPUComponent : public HybridGPUComponent {
 public:
  HybridGpsmPipelineGPUComponent(TraversalPlan* plan,
                                 TrackPartitionedGraph* rel,
                                 bool materialize_result, size_t thread_num)
      : HybridGPUComponent(plan, rel, materialize_result, thread_num),
        plan_(plan) {
    ProfilerHelper::AddGPUProfilePhaseGPSM(gpu_profiler_);
    ProfilerHelper::AddCountProfilePhaseGPSM(count_profiler_);
  }

  ~HybridGpsmPipelineGPUComponent() {
    for (size_t dev_id = 0; dev_id < plan_->GetDevPartitionNum(); ++dev_id) {
      CUDA_ERROR(cudaSetDevice(dev_id));
      delete dev_plans_[dev_id];
      dev_plans_[dev_id] = NULL;
      delete im_data_[dev_id];
      im_data_[dev_id] = NULL;
      delete im_data_holder_[dev_id];
      im_data_holder_[dev_id] = NULL;
    }
  }

  virtual WorkContext* CreateWorkContext(Task* task) {
    size_t d_partition_id = task->d_partition_id_;
    GpsmWorkContext* wctx = new GpsmWorkContext();
    wctx->Set(d_partition_id, thread_num_, cpu_relation_, task->ans_,
              cuda_contexts_[d_partition_id], d_partitions_[d_partition_id],
              gpu_profiler_, count_profiler_, plan_, im_data_[d_partition_id],
              im_data_holder_[d_partition_id], dev_plans_[d_partition_id]);
    return wctx;
  }

  virtual HybridGPUProcessor* CreateGPUProcessor(WorkContext* wctx,
                                                 Task* task) {
    GpsmWorkContext* gpsm_wctx = static_cast<GpsmWorkContext*>(wctx);
    return new HybridGPSMGPUProcessor(gpsm_wctx, task);
  }

  virtual void InitGPU() {
    dev_plans_.resize(plan_->GetDevPartitionNum());
    im_data_.resize(plan_->GetDevPartitionNum());
    im_data_holder_.resize(plan_->GetDevPartitionNum());
    for (size_t dev_id = 0; dev_id < plan_->GetDevPartitionNum(); ++dev_id) {
      CUDA_ERROR(cudaSetDevice(dev_id));
      dev_plans_[dev_id] = new DevTraversalPlan(plan_, cuda_contexts_[dev_id]);
      im_data_[dev_id] = new ImData(plan_);
      im_data_holder_[dev_id] =
          new ImDataDevHolder(plan_->GetVertexCount(), cuda_contexts_[dev_id]);
    }
  }

 private:
  TraversalPlan* plan_;
  std::vector<DevTraversalPlan*> dev_plans_;

 protected:
  std::vector<ImData*> im_data_;
  std::vector<ImDataDevHolder*> im_data_holder_;
};
}  // namespace Gpsm

#endif
