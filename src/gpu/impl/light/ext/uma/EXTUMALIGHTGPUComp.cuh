#ifndef __EXTERNAL_UMA_GPU_LIGHT_PIPELINE_COMPONENT_CUH__
#define __EXTERNAL_UMA_GPU_LIGHT_PIPELINE_COMPONENT_CUH__

#include "EXTLIGHTGPUComp.cuh"
#include "EXTUMALIGHTGPUProcessor.cuh"

namespace Light {
class EXTUMALIGHTPipelineGPUComponent : public EXTLIGHTPipelineGPUComponent {
 public:
  EXTUMALIGHTPipelineGPUComponent(LazyTraversalPlan *plan,
                                  TrackPartitionedGraph *rel, bool materialize,
                                  size_t thread_num)
      : plan_(plan),
        EXTLIGHTPipelineGPUComponent(plan, rel, materialize, thread_num, true) {
    // As incremental_load_subgraph_ is set to true, ReleaseBatch would not
    // release the graph
  }

  virtual HybridGPUProcessor *CreateGPUProcessor(WorkContext *wctx0,
                                                 Task *task) {
    LightWorkContext *wctx = static_cast<LightWorkContext *>(wctx0);
    return new EXTUMALIGHTGPUProcessor(wctx, task, incremental_load_subgraph_);
  }

  virtual void InitGPU() {
    EXTLIGHTPipelineGPUComponent::InitGPU();

    size_t partition_num = plan_->GetDevPartitionNum();
    dev_plan_.resize(partition_num);
    for (size_t dev_id = 0; dev_id < partition_num; ++dev_id) {
      CUDA_ERROR(cudaSetDevice(dev_id));
      dev_plan_[dev_id] =
          new DevLazyTraversalPlan(plan_, cuda_contexts_[dev_id]);

      // UMA allocate
      d_partitions_[dev_id]->UnTrackCopyHToD(cpu_relation_);
    }
  }

 private:
  LazyTraversalPlan *plan_;
  std::vector<DevLazyTraversalPlan *> dev_plan_;
};
}  // namespace Light

#endif