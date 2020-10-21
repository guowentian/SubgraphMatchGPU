#ifndef __HYBRID_GPU_LIGHT_INTERPLAY_PIPELINE_COMPONENT_CUH__
#define __HYBRID_GPU_LIGHT_INTERPLAY_PIPELINE_COMPONENT_CUH__

#include "EXTLIGHTGPUComp.cuh"

#include "HybridLIGHTItpGPUProcessor.cuh"

namespace Light {
class HybridLIGHTItpPipelineGPUComponent : public EXTLIGHTPipelineGPUComponent {
 public:
  HybridLIGHTItpPipelineGPUComponent(LazyTraversalPlan *plan,
                                     TrackPartitionedGraph *rel,
                                     bool materialize, size_t thread_num,
                                     bool incremental_load_subgraph = false)
      : EXTLIGHTPipelineGPUComponent(plan, rel, materialize, thread_num,
                                     incremental_load_subgraph),
        plan_(plan) {}

  ~HybridLIGHTItpPipelineGPUComponent() {
    size_t partition_num = group_dev_plan_.size();
    for (size_t dev_id = 0; dev_id < partition_num; ++dev_id) {
      CUDA_ERROR(cudaSetDevice(dev_id));

      for (auto &plan : group_dev_plan_[dev_id]) {
        delete plan;
        plan = NULL;
      }
      for (auto &plan : dfs_dev_plan_[dev_id]) {
        delete plan;
        plan = NULL;
      }
    }
  }

  virtual void GPUThreadExecute(Task *task) {
    assert(task->task_type_ == INTER_PARTITION);
    InterPartTask *inter_part_task = static_cast<InterPartTask *>(task);
    size_t d_partition_id = inter_part_task->d_partition_id_;
    auto context = cuda_contexts_[d_partition_id];

    CUDA_ERROR(cudaSetDevice(d_partition_id));
    gpu_profiler_->StartTimer("process_level_time", d_partition_id,
                              context->Stream());
    auto &group_dfs_ids = plan_->GetGroupDfsIds();
    for (size_t group_id = 0; group_id < group_dfs_ids.size(); ++group_id) {
      WorkContext *wctx = this->CreateWorkContextOnGroup(task, group_id);
      HybridGPUProcessor *processor = this->CreateGPUProcessor(wctx, task);
      processor->ProcessLevel(0);

      delete wctx;
      wctx = NULL;
      delete processor;
      processor = NULL;

      im_data_[d_partition_id]->Release();
      d_partitions_[d_partition_id]->Release();
    }
    gpu_profiler_->EndTimer("process_level_time", d_partition_id,
                            context->Stream());
  }

  WorkContext *CreateWorkContextOnGroup(Task *task, size_t group_id) {
    auto &group_dfs_ids = plan_->GetGroupDfsIds();
    size_t d_partition_id = task->d_partition_id_;
    LightItpWorkContext *wctx = new LightItpWorkContext();
    wctx->Set(d_partition_id, thread_num_, cpu_relation_, task->ans_,
              cuda_contexts_[d_partition_id], d_partitions_[d_partition_id],
              gpu_profiler_, count_profiler_, plan_, im_data_[d_partition_id],
              im_data_holder_[d_partition_id],
              group_dev_plan_[d_partition_id][group_id],
              graph_dev_trackers_[d_partition_id], group_id,
              group_dfs_ids[group_id][0], &dfs_dev_plan_);
    return wctx;
  }

  virtual HybridGPUProcessor *CreateGPUProcessor(WorkContext *wctx0,
                                                 Task *task) {
    LightItpWorkContext *wctx = static_cast<LightItpWorkContext *>(wctx0);
    return new HybridLIGHTItpGPUProcessor(wctx, task);
  }

  virtual void InitGPU() {
    EXTLIGHTPipelineGPUComponent::InitGPU();

    size_t vertex_count = plan_->GetVertexCount();
    AllCondType non_equality_cond(vertex_count);
    for (uintV u = 0; u < vertex_count; ++u) {
      for (uintV u2 = 0; u2 < vertex_count; ++u2) {
        if (u != u2) {
          non_equality_cond[u].push_back(std::make_pair(NON_EQUAL, u2));
        }
      }
    }

    size_t partition_num = plan_->GetDevPartitionNum();
    group_dev_plan_.resize(partition_num);
    dfs_dev_plan_.resize(partition_num);

    for (size_t dev_id = 0; dev_id < partition_num; ++dev_id) {
      CUDA_ERROR(cudaSetDevice(dev_id));

      auto &dfs_orders = plan_->GetSearchSequences();
      auto &group_dfs_ids = plan_->GetGroupDfsIds();
      size_t dfs_num = dfs_orders.size();
      size_t group_num = group_dfs_ids.size();
      size_t vertex_count = plan_->GetVertexCount();
      group_dev_plan_[dev_id].resize(group_num);
      dfs_dev_plan_[dev_id].resize(dfs_num);

      for (size_t group_id = 0; group_id < group_num; ++group_id) {
        size_t dfs_id = group_dfs_ids[group_id][0];
        auto &exec_seq = plan_->GetInterPartitionExecuteOperations()[dfs_id];

        AllConnType backward_conn;
        Plan::GetOrderedIndexBasedConnectivity(
            backward_conn, plan_->GetConnectivity(), dfs_orders[dfs_id]);
        AllCondType computed_cond;
        LazyTraversalPlanUtils::GetComputeCondition(
            computed_cond, exec_seq, non_equality_cond, vertex_count);
        AllCondType materialized_cond;
        LazyTraversalPlanUtils::GetMaterializeCondition(
            materialized_cond, exec_seq, non_equality_cond, vertex_count);
        AllCondType filter_cond;
        LazyTraversalPlanUtils::GetFilterCondition(
            filter_cond, exec_seq, non_equality_cond, vertex_count);
        AllCondType count_to_materialized_cond;
        LazyTraversalPlanUtils::GetCountToMaterializedVerticesCondition(
            count_to_materialized_cond, exec_seq, non_equality_cond,
            vertex_count);

        auto &materialized_vertices =
            plan_->GetInterPartitionMaterializedVertices()[dfs_id];
        auto &computed_unmaterialized_vertices =
            plan_->GetInterPartitionComputedUnmaterializedVertices()[dfs_id];

        group_dev_plan_[dev_id][group_id] = new DevLazyTraversalPlan(
            backward_conn, computed_cond, materialized_cond, filter_cond,
            count_to_materialized_cond, materialized_vertices,
            computed_unmaterialized_vertices, backward_conn, non_equality_cond,
            vertex_count, cuda_contexts_[dev_id]);
#if defined(DEBUG)
        std::cout << "====== group_id=" << group_id << "======" << std::endl;
        group_dev_plan_[dev_id][group_id]->Print();
#endif
      }

      for (size_t dfs_id = 0; dfs_id < dfs_num; ++dfs_id) {
        auto &exec_seq = plan_->GetInterPartitionExecuteOperations()[dfs_id];

        AllConnType backward_conn;
        Plan::GetOrderedIndexBasedConnectivity(
            backward_conn, plan_->GetConnectivity(), dfs_orders[dfs_id]);

        AllCondType index_based_order;
        Plan::GetIndexBasedOrdering(index_based_order, plan_->GetOrdering(),
                                    dfs_orders[dfs_id]);

        AllCondType computed_cond;
        LazyTraversalPlanUtils::GetComputeCondition(
            computed_cond, exec_seq, index_based_order, vertex_count);
        AllCondType materialized_cond;
        LazyTraversalPlanUtils::GetMaterializeCondition(
            materialized_cond, exec_seq, index_based_order, vertex_count);
        AllCondType filter_cond;
        LazyTraversalPlanUtils::GetFilterCondition(
            filter_cond, exec_seq, index_based_order, vertex_count);
        AllCondType count_to_materialized_cond;
        LazyTraversalPlanUtils::GetCountToMaterializedVerticesCondition(
            count_to_materialized_cond, exec_seq, index_based_order,
            vertex_count);

        auto &materialized_vertices =
            plan_->GetInterPartitionMaterializedVertices()[dfs_id];
        auto &computed_unmaterialized_vertices =
            plan_->GetInterPartitionComputedUnmaterializedVertices()[dfs_id];

        dfs_dev_plan_[dev_id][dfs_id] = new DevLazyTraversalPlan(
            backward_conn, computed_cond, materialized_cond, filter_cond,
            count_to_materialized_cond, materialized_vertices,
            computed_unmaterialized_vertices, backward_conn, index_based_order,
            vertex_count, cuda_contexts_[dev_id]);

#if defined(DEBUG)
        std::cout << "==== dfs_id=" << dfs_id << "====" << std::endl;
        dfs_dev_plan_[dev_id][dfs_id]->Print();
#endif
      }
    }
  }

 private:
  LazyTraversalPlan *plan_;

 protected:
  // each group has the dev_plan to enforce the connectivity and
  // non-equality constraints that are shared among dfs in this group
  std::vector<std::vector<DevLazyTraversalPlan *>> group_dev_plan_;
  // each dfs has separate ordering constraint used for counting
  std::vector<std::vector<DevLazyTraversalPlan *>> dfs_dev_plan_;
};  // namespace Light
}  // namespace Light

#endif
