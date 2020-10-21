#ifndef __HYBRID_GPU_LIGHT_INTERPLAY_PROCESSOR_CUH__
#define __HYBRID_GPU_LIGHT_INTERPLAY_PROCESSOR_CUH__

#include "EXTLIGHTGPUProcessor.cuh"
#include "LIGHTItpCommon.cuh"
#include "LIGHTItpComputeCount.cuh"
#include "LIGHTItpCount.cuh"
#include "LIGHTItpMaterialize.cuh"
#include "VerifyLIGHTItpCompute.cuh"

namespace Light {
class HybridLIGHTItpGPUProcessor : public EXTLIGHTGPUProcessor {
 public:
  HybridLIGHTItpGPUProcessor(LightItpWorkContext *wctx, Task *task)
      : EXTLIGHTGPUProcessor(wctx, task) {}

  virtual size_t GetNextLevel(size_t cur_exec_level) {
    if (cur_exec_level == 0) return cur_exec_level + 3;
    return cur_exec_level + 1;
  }

  virtual bool IsProcessOver(size_t cur_exec_level) {
    auto wctx = static_cast<LightItpWorkContext *>(wctx_);
    auto dfs_id = wctx->dfs_id;
    auto plan = wctx->plan;
    auto &exec_seq = plan->GetInterPartitionExecuteOperations()[dfs_id];
    return cur_exec_level == exec_seq.size();
  }

  virtual size_t GetLevelBatchSize(size_t cur_exec_level) {
    auto wctx = static_cast<LightItpWorkContext *>(wctx_);
    auto graph_partition = wctx->graph_partition;
    auto graph_dev_tracker = wctx->graph_dev_tracker;
    auto plan = wctx->plan;
    auto dfs_id = wctx->dfs_id;
    auto context = wctx->context;
    auto &exec_seq = plan->GetInterPartitionExecuteOperations()[dfs_id];

    size_t available_memory_size =
        context->GetDeviceMemoryInfo()->GetAvailableMemorySize();
    if (graph_partition->Empty()) {
      // For large graph, row_ptrs can take a large memory space, but the memory
      // estimation for OrganizeBatch does not consider this amount. So reserve
      // the memory for this amount if necessary.
      size_t graph_partition_reserve_size =
          sizeof(uintE) * graph_dev_tracker->GetVertexCount();
      assert(available_memory_size >= graph_partition_reserve_size);
      available_memory_size -= graph_partition_reserve_size;
    }
    size_t remaining_levels_num = exec_seq.size() - cur_exec_level;
    size_t P = available_memory_size / remaining_levels_num;
    return BatchManager::GetSafeBatchSize(P);
  }

  virtual void BeforeBatchProcess(size_t cur_exec_level,
                                  BatchData *&batch_data) {
    auto wctx = static_cast<LightItpWorkContext *>(wctx_);
    auto dfs_id = wctx->dfs_id;
    auto plan = wctx->plan;
    auto im_data = wctx->im_data;
    auto &exec_seq = plan->GetInterPartitionExecuteOperations()[dfs_id];
    auto &materialized_vertices =
        plan->GetInterPartitionMaterializedVertices()[dfs_id][cur_exec_level];
    auto &computed_unmaterialized_vertices =
        plan->GetInterPartitionComputedUnmaterializedVertices()[dfs_id]
                                                               [cur_exec_level];
    ImResult *im_result =
        new ImResult(im_data, materialized_vertices,
                     computed_unmaterialized_vertices, plan->GetVertexCount());
    auto im_batch_data = new ImBatchData();
    im_batch_data->im_result = im_result;
    batch_data = im_batch_data;
  }

  virtual void AfterBatchProcess(size_t cur_exec_level,
                                 BatchData *&batch_data) {
    auto wctx = static_cast<LightItpWorkContext *>(wctx_);
    auto im_batch_data = static_cast<ImBatchData *>(batch_data);
    auto dfs_id = wctx->dfs_id;
    auto plan = wctx->plan;
    auto im_data = wctx->im_data;
    auto &exec_seq = plan->GetInterPartitionExecuteOperations()[dfs_id];
    auto &materialized_vertices =
        plan->GetInterPartitionMaterializedVertices()[dfs_id][cur_exec_level];
    auto &computed_unmaterialized_vertices =
        plan->GetInterPartitionComputedUnmaterializedVertices()[dfs_id]
                                                               [cur_exec_level];
    auto &im_result = im_batch_data->im_result;
    im_result->Swap(im_data, materialized_vertices,
                    computed_unmaterialized_vertices, plan->GetVertexCount());
    delete im_result;
    im_result = NULL;
    delete im_batch_data;
    im_batch_data = NULL;
  }

  virtual void PrepareBatch(size_t cur_exec_level, BatchData *batch_data,
                            BatchSpec batch_spec) {
    if (cur_exec_level == 0) {
      return;
    }
    auto wctx = static_cast<LightItpWorkContext *>(wctx_);
    auto im_batch_data = static_cast<ImBatchData *>(batch_data);
    auto dfs_id = wctx->dfs_id;
    auto plan = wctx->plan;
    auto im_data = wctx->im_data;
    auto dev_plan = wctx->dev_plan;
    auto im_result = im_batch_data->im_result;

    auto &exec_seq = plan->GetInterPartitionExecuteOperations()[dfs_id];
    auto &materialized_vertices =
        plan->GetInterPartitionMaterializedVertices()[dfs_id][cur_exec_level];
    auto &computed_unmaterialized_vertices =
        plan->GetInterPartitionComputedUnmaterializedVertices()[dfs_id]
                                                               [cur_exec_level];
    auto exec_seq_op = exec_seq[cur_exec_level].first;

    im_data->CopyBatchData(im_result, &batch_spec, materialized_vertices,
                           computed_unmaterialized_vertices);

    // obtain the external graph if necessary
    if (exec_seq_op == COMPUTE || exec_seq_op == COMPUTE_COUNT) {
      DevConnType *d_conn = dev_plan->GetBackwardConnectivity()->GetArray() +
                            exec_seq[cur_exec_level].second;
      auto &d_instances = im_data->GetInstances();
      size_t path_num = d_instances[materialized_vertices[0]]->GetSize();
      LoadSubgraph(wctx, d_conn, path_num);
    }
  }

  virtual void ReleaseBatch(size_t cur_exec_level, BatchData *batch_data,
                            BatchSpec batch_spec) {
    auto wctx = static_cast<LightItpWorkContext *>(wctx_);
    auto im_data = wctx->im_data;
    auto graph_partition = wctx->graph_partition;

    im_data->Release();
    if (!incremental_load_subgraph_) {
      graph_partition->Release();
    }
  }

  virtual void PrintProgress(size_t cur_exec_level, size_t batch_id,
                             size_t batch_num, BatchSpec batch_spec) {
#if defined(PROFILE)
    auto wctx = static_cast<LightItpWorkContext *>(wctx_);
    auto dfs_group_id = wctx->dfs_group_id;
    auto dfs_id = wctx->dfs_id;
    std::cout << "dfs_group_id=" << dfs_group_id << ",dfs_id=" << dfs_id << ",";
    EXTLIGHTGPUProcessor::PrintProgress(cur_exec_level, batch_id, batch_num,
                                        batch_spec);
#endif
  }

  virtual void CountInterPart(LightItpWorkContext *wctx,
                              size_t cur_exec_level) {
    auto dfs_group_id = wctx->dfs_group_id;
    auto d_partition_id = wctx->d_partition_id;
    auto plan = wctx->plan;
    auto &ans = *wctx->ans;
    auto &dfs_dev_plan = *wctx->dfs_dev_plan;

    bool enable_ordering = plan->GetQuery()->GetEnableOrdering();
    auto &dfs_group = plan->GetGroupDfsIds()[dfs_group_id];
    for (auto dfs_id : dfs_group) {
      LightItpWorkContext nwctx;
      nwctx = *wctx;
      nwctx.dev_plan = dfs_dev_plan[d_partition_id][dfs_id];

      size_t ret = ItpLIGHTCount(&nwctx, cur_exec_level);
      if (enable_ordering) {
        ans += ret;
      } else {
        ans += ret * dfs_group.size();
        break;
      }
    }
  }

  virtual void ComputeCountInterPart(LightItpWorkContext *wctx,
                                     size_t cur_exec_level) {
    auto dfs_group_id = wctx->dfs_group_id;
    auto d_partition_id = wctx->d_partition_id;
    auto plan = wctx->plan;
    auto &ans = *wctx->ans;
    auto &dfs_dev_plan = *wctx->dfs_dev_plan;

    bool enable_ordering = plan->GetQuery()->GetEnableOrdering();
    auto &dfs_group = plan->GetGroupDfsIds()[dfs_group_id];
    for (auto dfs_id : dfs_group) {
      LightItpWorkContext nwctx;
      nwctx = *wctx;
      nwctx.dev_plan = dfs_dev_plan[d_partition_id][dfs_id];

      size_t ret = ItpComputeCount(&nwctx, cur_exec_level);
      if (enable_ordering) {
        ans += ret;
      } else {
        ans += ret * dfs_group.size();
        break;
      }
    }
  }

  virtual void OrganizeBatch(size_t cur_exec_level,
                             BatchManager *batch_manager) {
    auto wctx = static_cast<LightItpWorkContext *>(wctx_);
    auto task = static_cast<InterPartTask *>(task_);
    auto dfs_id = wctx->dfs_id;
    auto d_partition_id = wctx->d_partition_id;
    auto context = wctx->context;
    auto plan = wctx->plan;
    auto im_data = wctx->im_data;
    auto im_data_holder = wctx->im_data_holder;
    auto dev_plan = wctx->dev_plan;
    auto graph_dev_tracker = wctx->graph_dev_tracker;

    auto &exec_seq = plan->GetInterPartitionExecuteOperations()[dfs_id];
    auto &materialized_vertices =
        plan->GetInterPartitionMaterializedVertices()[dfs_id][cur_exec_level];
    auto &computed_unmaterialized_vertices =
        plan->GetInterPartitionComputedUnmaterializedVertices()[dfs_id]
                                                               [cur_exec_level];
    if (cur_exec_level == 0) {
      // d_instances[0] + d_instances[1]
      size_t parent_factor = sizeof(uintV) * 2;
      size_t remaining_levels_num = exec_seq.size() - cur_exec_level;
      size_t temporary_parent_factor =
          std::ceil(1.0 * sizeof(size_t) * 3 /
                    remaining_levels_num);  // OrganizeBatch in next level
      parent_factor += temporary_parent_factor;
      batch_manager->OrganizeBatch(task->GetEdgeCount(), parent_factor);

      return;
    }

    auto op = exec_seq[cur_exec_level].first;
    uintV cur_level = exec_seq[cur_exec_level].second;
    size_t path_num =
        im_data->GetInstances()[materialized_vertices[0]]->GetSize();

    switch (op) {
      case COMPUTE:
      case COMPUTE_COUNT: {
        DeviceArray<size_t> *children_count = NULL;
        DeviceArray<size_t> *children_cost = NULL;
        size_t parent_factor = 0;
        size_t children_factor = 0;

        if (op == COMPUTE) {
          EstimateComputeMemoryCost(exec_seq, cur_exec_level, parent_factor,
                                    children_factor);
        } else {
          EstimateComputeCountMemoryCost(exec_seq, cur_exec_level,
                                         parent_factor, children_factor);
        }
        EXTEstimateIntersectCost(wctx, cur_level, path_num, parent_factor,
                                 children_factor, children_count,
                                 children_cost);

        batch_manager->OrganizeBatch(children_count, children_cost,
                                     children_count->GetSize(), context);
        // combine the batches if they do not exceed the batch size

        uintV **d_seq_instances = im_data_holder->GetSeqInstances()->GetArray();
        DevConnType *conn =
            dev_plan->GetBackwardConnectivity()->GetArray() + cur_level;
        BackwardNeighborGatherFunctor backward_neighbor_functor(
            conn, d_seq_instances);
        CombineLoadSubgraphBatch(d_partition_id, context,
                                 backward_neighbor_functor, graph_dev_tracker,
                                 batch_manager, parent_factor, children_factor,
                                 children_count);

        delete children_count;
        children_count = NULL;
        delete children_cost;
        children_cost = NULL;
      } break;
      case MATERIALIZE:
      case FILTER_COMPUTE: {
        DeviceArray<size_t> *children_count = NULL;
        size_t parent_factor = 0;
        size_t children_factor = 0;
        if (op == FILTER_COMPUTE) {
          EstimateFilterComputeMemoryCost(exec_seq, cur_exec_level,
                                          parent_factor, children_factor);

        } else {
          EstimateMaterializeMemoryCost(
              exec_seq, materialized_vertices, computed_unmaterialized_vertices,
              cur_exec_level, parent_factor, children_factor);
        }
        BuildGatherChildrenCount(wctx, cur_level, path_num, children_count);

        batch_manager->OrganizeBatch(children_count, parent_factor,
                                     children_factor, children_count->GetSize(),
                                     context);

        delete children_count;
        children_count = NULL;
      } break;
      case COUNT: {
        DeviceArray<size_t> *children_count = NULL;
        size_t parent_factor = 0;
        size_t children_factor = 0;
        if (computed_unmaterialized_vertices.size() == 1) {
          // for valid_path_ids
          parent_factor = sizeof(size_t);
          children_factor = 0;
        } else {
          // valid_path_ids, filter_candidates_offsets, workload_offsets
          parent_factor =
              sizeof(size_t) +
              computed_unmaterialized_vertices.size() * sizeof(size_t) +
              sizeof(size_t);
          // filter_candidates, path_ids
          children_factor =
              sizeof(uintV) * computed_unmaterialized_vertices.size() +
              sizeof(size_t);
        }
        // TODO: chlidren_count may not be necessary when
        // computed_unmaterialized_vertices.size() == 1
        BuildGatherChildrenCount(wctx, computed_unmaterialized_vertices[0],
                                 path_num, children_count);

        batch_manager->OrganizeBatch(children_count, parent_factor,
                                     children_factor, children_count->GetSize(),
                                     context);

        ReleaseIfExists(children_count);
      } break;
      default:
        break;
    }
  }

  virtual void ExecuteBatch(size_t cur_exec_level, BatchSpec batch_spec) {
    auto wctx = static_cast<LightItpWorkContext *>(wctx_);
    auto task = static_cast<InterPartTask *>(task_);
    auto dfs_id = wctx->dfs_id;
    auto cpu_relation = wctx->cpu_relation;
    auto d_partition_id = wctx->d_partition_id;
    auto context = wctx->context;
    auto plan = wctx->plan;
    auto im_data = wctx->im_data;
    auto gpu_profiler = wctx->gpu_profiler;

    auto &exec_seq = plan->GetInterPartitionExecuteOperations()[dfs_id];
    auto op = exec_seq[cur_exec_level].first;
    uintV cur_level = exec_seq[cur_exec_level].second;

    if (cur_exec_level == 0) {
      assert(exec_seq[0].first == MATERIALIZE && exec_seq[1].first == COMPUTE &&
             exec_seq[2].first == MATERIALIZE);
      assert(exec_seq[0].second == 0 && exec_seq[1].second == 1 &&
             exec_seq[2].second == 1);

      InitFirstTwoLevel(wctx, cpu_relation, task, exec_seq[0].second,
                        exec_seq[1].second, &batch_spec);
      return;
    }

    auto &materialized_vertices =
        plan->GetInterPartitionMaterializedVertices()[dfs_id][cur_exec_level];
    auto &computed_unmaterialized_vertices =
        plan->GetInterPartitionComputedUnmaterializedVertices()[dfs_id]
                                                               [cur_exec_level];
    auto &d_instances = im_data->GetInstances();
    size_t path_num = d_instances[materialized_vertices[0]]->GetSize();

    switch (op) {
      case COMPUTE: {
        gpu_profiler->StartTimer("compute_time", d_partition_id,
                                 context->Stream());
        ComputeGeneral<true>(wctx, cur_level, path_num);
        gpu_profiler->EndTimer("compute_time", d_partition_id,
                               context->Stream());
#if defined(DEBUG)
        /*VerifyItpCompute(d_partition_id, dfs_id, cur_exec_level, context,
           im_data, im_data_holder, dev_plan, graph_partition, plan,
                         graph_dev_tracker, cpu_relation_, task, ans);
                         */
#endif
      } break;
      case FILTER_COMPUTE: {
        gpu_profiler->StartTimer("filter_compute_time", d_partition_id,
                                 context->Stream());
        FilterCompute(wctx, cur_level);
        gpu_profiler->EndTimer("filter_compute_time", d_partition_id,
                               context->Stream());

      } break;
      case MATERIALIZE: {
        gpu_profiler->StartTimer("materialize_time", d_partition_id,
                                 context->Stream());
        Materialize(wctx, cur_level, materialized_vertices,
                    computed_unmaterialized_vertices);

        gpu_profiler->EndTimer("materialize_time", d_partition_id,
                               context->Stream());
      } break;
      case COMPUTE_COUNT: {
        gpu_profiler->StartTimer("compute_count_time", d_partition_id,
                                 context->Stream());
        ComputeCountInterPart(wctx, cur_exec_level);
        gpu_profiler->EndTimer("compute_count_time", d_partition_id,
                               context->Stream());
      } break;
      case COUNT: {
        gpu_profiler->StartTimer("count_time", d_partition_id,
                                 context->Stream());
        CountInterPart(wctx, cur_exec_level);
        gpu_profiler->EndTimer("count_time", d_partition_id, context->Stream());
      } break;
      default:
        assert(false);
        break;
    }
  }
  virtual bool NeedSearchNext(size_t cur_exec_level) {
    auto wctx = static_cast<LightItpWorkContext *>(wctx_);
    auto dfs_id = wctx->dfs_id;
    auto plan = wctx->plan;
    auto im_data = wctx->im_data;
    auto &exec_seq = plan->GetInterPartitionExecuteOperations()[dfs_id];
    auto &materialized_vertices =
        plan->GetInterPartitionMaterializedVertices()[dfs_id][cur_exec_level];

    if (cur_exec_level == 0) {
      return true;
    }
    bool ret = false;
    switch (exec_seq[cur_exec_level].first) {
      case COMPUTE:
      case FILTER_COMPUTE: {
        auto cur_level = exec_seq[cur_exec_level].second;
        ret = im_data->GetCandidates()[cur_level]->GetSize() > 0;
      } break;
      case MATERIALIZE:
      case COMPUTE_PATH_COUNT:
        ret = im_data->GetInstances()[materialized_vertices[0]]->GetSize() > 0;
        break;
      case COMPUTE_COUNT:
      case COUNT:
        ret = false;
        break;
      default:
        break;
    }
    return ret;
  }
  virtual void CollectCount(size_t cur_exec_level) {
#if defined(PROFILE)
    auto wctx = static_cast<LightItpWorkContext *>(wctx_);
    auto d_partition_id = wctx->d_partition_id;
    auto dfs_id = wctx->dfs_id;
    auto plan = wctx->plan;
    auto im_data = wctx->im_data;
    auto &exec_seq = plan->GetInterPartitionExecuteOperations()[dfs_id];
    auto &materialized_vertices =
        plan->GetInterPartitionMaterializedVertices()[dfs_id][cur_exec_level];

    auto count_profiler = wctx->count_profiler;
    count_profiler->AddCount("generate_batch_count", d_partition_id, 1);

    if (cur_exec_level > 0) {
      size_t path_num =
          im_data->GetInstances()[materialized_vertices[0]]->GetSize();
      switch (exec_seq[cur_exec_level].first) {
        case COMPUTE:
        case COMPUTE_COUNT:
        case COMPUTE_PATH_COUNT:
          count_profiler->AddCount("compute_count", d_partition_id, path_num);
          break;
        case MATERIALIZE:
          count_profiler->AddCount("materialize_count", d_partition_id,
                                   path_num);
          break;
        default:
          break;
      }
    }
#endif
  }
};
}  // namespace Light

#endif