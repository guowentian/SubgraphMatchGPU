#ifndef __EXTERNAL_GPU_LIGHT_PIPELINE_COMPONENT_CUH__
#define __EXTERNAL_GPU_LIGHT_PIPELINE_COMPONENT_CUH__

#include "EXTLIGHTGPUProcessor.cuh"
#include "HybridLIGHTGPUComp.cuh"

namespace Light {
class EXTLIGHTPipelineGPUComponent : public HybridLIGHTPipelineGPUComponent {
 public:
  EXTLIGHTPipelineGPUComponent(LazyTraversalPlan *plan,
                               TrackPartitionedGraph *rel, bool materialize,
                               size_t thread_num,
                               bool incremental_load_subgraph = false)
      : HybridLIGHTPipelineGPUComponent(plan, rel, materialize, thread_num),
        plan_(plan),
        incremental_load_subgraph_(incremental_load_subgraph) {
    ProfilerHelper::AddGPUProfilePhaseEXT(gpu_profiler_);
    ProfilerHelper::AddCountProfilePhaseEXT(count_profiler_);
  }
  ~EXTLIGHTPipelineGPUComponent() {
    size_t p = plan_->GetDevPartitionNum();
    for (size_t dev_id = 0; dev_id < p; ++dev_id) {
      CUDA_ERROR(cudaSetDevice(dev_id));
      delete dev_plan_[dev_id];
      dev_plan_[dev_id] = NULL;
      delete graph_dev_trackers_[dev_id];
      graph_dev_trackers_[dev_id] = NULL;
    }
  }

  virtual WorkContext *CreateWorkContext(Task *task) {
    LightWorkContext *wctx = new LightWorkContext();
    size_t d_partition_id = task->d_partition_id_;
    wctx->Set(d_partition_id, thread_num_, cpu_relation_, task->ans_,
              cuda_contexts_[d_partition_id], d_partitions_[d_partition_id],
              gpu_profiler_, count_profiler_, plan_, im_data_[d_partition_id],
              im_data_holder_[d_partition_id], dev_plan_[d_partition_id],
              graph_dev_trackers_[d_partition_id]);
    return wctx;
  }

  virtual HybridGPUProcessor *CreateGPUProcessor(WorkContext *wctx0,
                                                 Task *task) {
    LightWorkContext *wctx = static_cast<LightWorkContext *>(wctx0);
    return new EXTLIGHTGPUProcessor(wctx, task, incremental_load_subgraph_);
  }

  virtual void InitGPU() {
    HybridLIGHTPipelineGPUComponent::InitGPU();

    size_t partition_num = plan_->GetDevPartitionNum();
    dev_plan_.resize(partition_num);
    graph_dev_trackers_.resize(partition_num);
    for (size_t dev_id = 0; dev_id < partition_num; ++dev_id) {
      CUDA_ERROR(cudaSetDevice(dev_id));
      dev_plan_[dev_id] =
          new DevLazyTraversalPlan(plan_, cuda_contexts_[dev_id]);
      graph_dev_trackers_[dev_id] = new GraphDevTracker(
          cpu_relation_->GetVertexPartitionMap(), cpu_relation_->GetRowPtrs(),
          cpu_relation_->GetVertexCount(), cpu_relation_->GetEdgeCount(),
          cuda_contexts_[dev_id]);
    }
  }

 private:
  LazyTraversalPlan *plan_;
  std::vector<DevLazyTraversalPlan *> dev_plan_;

 protected:
  std::vector<GraphDevTracker *> graph_dev_trackers_;

  // A flag to control the execution to load the subgraph in an incremental
  // way. Specifically, when the adjacent list required is in GPU memory
  // already, we do not load it from main memory. This can save communication
  // cost but increase overhead of graph management in GPUs.
  bool incremental_load_subgraph_;
};
}  // namespace Light

#endif
