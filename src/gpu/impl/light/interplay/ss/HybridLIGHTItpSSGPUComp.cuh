#ifndef __HYBRID_LIGHT_INTERPLAY_PIPELINE_GPU_SINGLE_SEARCH_SEQUENCE_CUH__
#define __HYBRID_LIGHT_INTERPLAY_PIPELINE_GPU_SINGLE_SEARCH_SEQUENCE_CUH__

#include "HybridLIGHTItpGPUComp.cuh"
#include "HybridLIGHTItpSSGPUProcessor.cuh"

namespace Light {
// GPU component searches inter-partition instances:
// for each search sequence, we search the inter-partition instances separately
// without the grouping optimization
class HybridLIGHTItpSingleSequenceGPUComponent
    : public HybridLIGHTItpPipelineGPUComponent {
 public:
  HybridLIGHTItpSingleSequenceGPUComponent(LazyTraversalPlan *plan,
                                           TrackPartitionedGraph *rel,
                                           bool materialize, size_t thread_num)
      : HybridLIGHTItpPipelineGPUComponent(plan, rel, materialize, true,
                                           thread_num),
        plan_(plan) {}
  ~HybridLIGHTItpSingleSequenceGPUComponent() {}

  virtual void GPUThreadExecute(Task *task) {
    assert(task->task_type_ == INTER_PARTITION);
    size_t d_partition_id = task->d_partition_id_;
    auto context = cuda_contexts_[d_partition_id];

    CUDA_ERROR(cudaSetDevice(d_partition_id));
    gpu_profiler_->StartTimer("process_level_time", d_partition_id,
                              context->Stream());
    auto &group_dfs_ids = plan_->GetGroupDfsIds();
    for (size_t group_id = 0; group_id < group_dfs_ids.size(); ++group_id) {
      for (auto dfs_id : group_dfs_ids[group_id]) {
        WorkContext *wctx =
            this->CreateWorkContextOnDfs(task, group_id, dfs_id);
        HybridGPUProcessor *processor = this->CreateGPUProcessor(wctx, task);
        processor->ProcessLevel(0);

        delete wctx;
        wctx = NULL;
        delete processor;
        processor = NULL;

        im_data_[d_partition_id]->Release();
        d_partitions_[d_partition_id]->Release();
      }
    }
    gpu_profiler_->EndTimer("process_level_time", d_partition_id,
                            context->Stream());
  }

  WorkContext *CreateWorkContextOnDfs(Task *task, size_t group_id,
                                      size_t dfs_id) {
    LightItpWorkContext *wctx = new LightItpWorkContext();
    auto &group_dfs_ids = plan_->GetGroupDfsIds();
    size_t d_partition_id = task->d_partition_id_;
    wctx->Set(
        d_partition_id, thread_num_, cpu_relation_, task->ans_,
        cuda_contexts_[d_partition_id], d_partitions_[d_partition_id],
        gpu_profiler_, count_profiler_, plan_, im_data_[d_partition_id],
        im_data_holder_[d_partition_id], dfs_dev_plan_[d_partition_id][dfs_id],
        graph_dev_trackers_[d_partition_id], group_id, dfs_id, &dfs_dev_plan_);
    return wctx;
  }

  virtual HybridGPUProcessor *CreateGPUProcessor(WorkContext *wctx0,
                                                 Task *task0) {
    LightItpWorkContext *wctx = static_cast<LightItpWorkContext *>(wctx0);
    InterPartTask *task = static_cast<InterPartTask *>(task0);
    return new HybridLIGHTItpSSGPUProcessor(wctx, task);
  }

 private:
  LazyTraversalPlan *plan_;
};
}  // namespace Light

#endif
