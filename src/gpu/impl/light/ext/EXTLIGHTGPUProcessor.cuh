#ifndef __EXTERNAL_LIGHT_GPU_PROCESSOR_CUH__
#define __EXTERNAL_LIGHT_GPU_PROCESSOR_CUH__

#include "EXTBatchManager.cuh"
#include "EXTGraphGPUCommon.cuh"
#include "EXTIncGraphGPUCommon.cuh"
#include "EXTLIGHTCommon.cuh"
#include "EXTLIGHTOrganizeBatch.cuh"
#include "GraphDevTracker.cuh"
#include "HybridLIGHTGPUProcessor.cuh"

namespace Light {
class EXTLIGHTGPUProcessor : public HybridLIGHTGPUProcessor {
 public:
  EXTLIGHTGPUProcessor(LightWorkContext *wctx, Task *task,
                       bool incremental_load_subgraph = false)
      : HybridLIGHTGPUProcessor(wctx, task),
        incremental_load_subgraph_(incremental_load_subgraph) {}

  virtual void PrepareBatch(size_t cur_exec_level, BatchData *im_batch_data,
                            BatchSpec batch_spec) {
    HybridLIGHTGPUProcessor::PrepareBatch(cur_exec_level, im_batch_data,
                                          batch_spec);
    LightWorkContext *wctx = static_cast<LightWorkContext *>(wctx_);

    // obtain the external graph if necessary
    auto im_data = wctx->im_data;
    auto plan = wctx->plan;
    auto &exec_seq = plan->GetExecuteOperations();
    auto op = exec_seq[cur_exec_level].first;
    if (op == COMPUTE || op == COMPUTE_PATH_COUNT || op == COMPUTE_COUNT) {
      auto dev_plan = wctx->dev_plan;
      auto &materialized_vertices =
          plan->GetMaterializedVertices()[cur_exec_level];

      DevConnType *d_conn = dev_plan->GetBackwardConnectivity()->GetArray() +
                            exec_seq[cur_exec_level].second;
      auto &d_instances = im_data->GetInstances();
      size_t path_num = d_instances[materialized_vertices[0]]->GetSize();
      LoadSubgraph(wctx, d_conn, path_num);
    }
  }
  virtual void ReleaseBatch(size_t cur_exec_level, BatchData *im_batch_data,
                            BatchSpec batch_spec) {
    HybridLIGHTGPUProcessor::ReleaseBatch(cur_exec_level, im_batch_data,
                                          batch_spec);

    if (!incremental_load_subgraph_) {
      LightWorkContext *wctx = static_cast<LightWorkContext *>(wctx_);
      auto graph_partition = wctx->graph_partition;
      graph_partition->Release();
    }
  }
  virtual void LoadSubgraph(LightWorkContext *wctx, DevConnType *conn,
                            size_t path_num) {
    auto d_partition_id = wctx->d_partition_id;
    auto context = wctx->context;
    auto cpu_relation = wctx->cpu_relation;
    auto thread_num = wctx->thread_num;
    auto im_data = wctx->im_data;
    auto im_data_holder = wctx->im_data_holder;
    auto graph_partition = wctx->graph_partition;
    auto graph_dev_tracker = wctx->graph_dev_tracker;
    auto gpu_profiler = wctx->gpu_profiler;
    auto count_profiler = wctx->count_profiler;

    // After CopyBatchData, we should update d_seq_instances
    im_data_holder->GatherImData(im_data, context);
    uintV **d_seq_instances = im_data_holder->GetSeqInstances()->GetArray();
    BackwardNeighborGatherFunctor backward_neighbor_functor(conn,
                                                            d_seq_instances);
    if (incremental_load_subgraph_) {
      if (graph_partition->Empty()) {
        BlockingLoadSubgraph(d_partition_id, context, graph_partition,
                             graph_dev_tracker, cpu_relation,
                             backward_neighbor_functor, path_num, gpu_profiler,
                             count_profiler, thread_num);
      } else {
        IncBlockingLoadSubgraph(d_partition_id, context, graph_partition,
                                graph_dev_tracker, cpu_relation,
                                backward_neighbor_functor, path_num,
                                gpu_profiler, count_profiler, thread_num);
      }
    } else {
      BlockingLoadSubgraph(d_partition_id, context, graph_partition,
                           graph_dev_tracker, cpu_relation,
                           backward_neighbor_functor, path_num, gpu_profiler,
                           count_profiler, thread_num);
    }
  }

  virtual void OrganizeBatch(size_t cur_exec_level,
                             BatchManager *batch_manager) {
    LightWorkContext *wctx = static_cast<LightWorkContext *>(wctx_);
    auto plan = wctx->plan;
    auto &exec_seq = plan->GetExecuteOperations();
    auto op = exec_seq[cur_exec_level].first;
    if (cur_exec_level == 0 ||
        (op == FILTER_COMPUTE || op == MATERIALIZE || op == COUNT)) {
      HybridLIGHTGPUProcessor::OrganizeBatch(cur_exec_level, batch_manager);
      return;
    }

    auto d_partition_id = wctx->d_partition_id;
    auto context = wctx->context;
    auto im_data = wctx->im_data;
    auto im_data_holder = wctx->im_data_holder;
    auto dev_plan = wctx->dev_plan;
    auto graph_dev_tracker = wctx->graph_dev_tracker;

    auto &materialized_vertices =
        plan->GetMaterializedVertices()[cur_exec_level];
    auto &computed_unmaterialized_vertices =
        plan->GetComputedUnmaterializedVertices()[cur_exec_level];

    uintV cur_level = exec_seq[cur_exec_level].second;
    size_t path_num =
        im_data->GetInstances()[materialized_vertices[0]]->GetSize();

    // Estimate parent_factor and children_factor
    size_t parent_factor = 0;
    size_t children_factor = 0;

    // We do not consider the following cost in our estimation,
    // because we expect they would not cause the memory overflow
    //
    // InspectJoin: bitmaps, bool,
    // |V| all_vertex_ids, uintV, |V|

    // BuildSubgraph:
    // required_row_ptrs, required_vertex_ids

    switch (op) {
      case COMPUTE:
        EstimateComputeMemoryCost(exec_seq, cur_exec_level, parent_factor,
                                  children_factor);
        break;
      case COMPUTE_COUNT:
        EstimateComputeCountMemoryCost(exec_seq, cur_exec_level, parent_factor,
                                       children_factor);
        break;
      case COMPUTE_PATH_COUNT:
        EstimateComputePathCountMemoryCost(exec_seq, cur_exec_level,
                                           parent_factor, children_factor);
        break;
      default:
        assert(false);
        break;
    }

    DeviceArray<size_t> *children_count = NULL;
    DeviceArray<size_t> *children_cost = NULL;
    EXTEstimateIntersectCost(wctx, cur_level, path_num, parent_factor,
                             children_factor, children_count, children_cost);

    batch_manager->OrganizeBatch(children_count, children_cost,
                                 children_count->GetSize(), context);

    uintV **d_seq_instances = im_data_holder->GetSeqInstances()->GetArray();
    DevConnType *conn =
        dev_plan->GetBackwardConnectivity()->GetArray() + cur_level;
    BackwardNeighborGatherFunctor backward_neighbor_functor(conn,
                                                            d_seq_instances);
    CombineLoadSubgraphBatch(d_partition_id, context, backward_neighbor_functor,
                             graph_dev_tracker, batch_manager, parent_factor,
                             children_factor, children_count);

    delete children_cost;
    children_cost = NULL;
    delete children_count;
    children_count = NULL;
  }

 protected:
  bool incremental_load_subgraph_;
};
}  // namespace Light

#endif
