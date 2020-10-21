#ifndef __HYBRID_LIGHT_GPU_PROCESSOR_CUH__
#define __HYBRID_LIGHT_GPU_PROCESSOR_CUH__

#include "DevLazyTraversalPlan.cuh"
#include "HybridGPUProcessor.cuh"
#include "LIGHTCommon.cuh"
#include "LIGHTCompute.cuh"
#include "LIGHTComputeCount.cuh"
#include "LIGHTComputePathCount.cuh"
#include "LIGHTCount.cuh"
#include "LIGHTDirectCount.cuh"
#include "LIGHTFilterCompute.cuh"
#include "LIGHTMaterialize.cuh"
#include "LIGHTOrganizeBatch.cuh"

namespace Light {
class HybridLIGHTGPUProcessor : public HybridGPUProcessor {
 public:
  HybridLIGHTGPUProcessor(LightWorkContext *wctx, Task *task)
      : HybridGPUProcessor(wctx, task) {}

  virtual bool IsProcessOver(size_t cur_exec_level) {
    auto wctx = static_cast<LightWorkContext *>(wctx_);
    auto plan = wctx->plan;
    auto &exec_seq = plan->GetExecuteOperations();
    return cur_exec_level == exec_seq.size();
  }

  virtual size_t GetLevelBatchSize(size_t cur_exec_level) {
    auto wctx = static_cast<LightWorkContext *>(wctx_);
    auto plan = wctx->plan;
    auto context = wctx->context;
    auto &exec_seq = plan->GetExecuteOperations();
    size_t remaining_levels_num = exec_seq.size() - cur_exec_level;
    size_t P = context->GetDeviceMemoryInfo()->GetAvailableMemorySize() /
               remaining_levels_num;
    return BatchManager::GetSafeBatchSize(P);
  }

  virtual void BeforeBatchProcess(size_t cur_exec_level,
                                  BatchData *&batch_data) {
    auto wctx = static_cast<LightWorkContext *>(wctx_);
    auto plan = wctx->plan;
    auto im_data = wctx->im_data;
    auto &exec_seq = plan->GetExecuteOperations();
    auto &materialized_vertices =
        plan->GetMaterializedVertices()[cur_exec_level];
    auto &computed_unmaterialized_vertices =
        plan->GetComputedUnmaterializedVertices()[cur_exec_level];

    ImResult *im_result =
        new ImResult(im_data, materialized_vertices,
                     computed_unmaterialized_vertices, plan->GetVertexCount());
    auto im_batch_data = new ImBatchData();
    im_batch_data->im_result = im_result;
    batch_data = im_batch_data;
  }

  virtual void AfterBatchProcess(size_t cur_exec_level,
                                 BatchData *&batch_data) {
    auto wctx = static_cast<LightWorkContext *>(wctx_);
    auto im_batch_data = static_cast<ImBatchData *>(batch_data);
    auto plan = wctx->plan;
    auto im_data = wctx->im_data;
    auto &exec_seq = plan->GetExecuteOperations();
    auto &materialized_vertices =
        plan->GetMaterializedVertices()[cur_exec_level];
    auto &computed_unmaterialized_vertices =
        plan->GetComputedUnmaterializedVertices()[cur_exec_level];

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
    auto wctx = static_cast<LightWorkContext *>(wctx_);
    auto im_batch_data = static_cast<ImBatchData *>(batch_data);
    auto im_data = wctx->im_data;
    auto plan = wctx->plan;
    auto &materialized_vertices =
        plan->GetMaterializedVertices()[cur_exec_level];
    auto &computed_unmaterialized_vertices =
        plan->GetComputedUnmaterializedVertices()[cur_exec_level];
    auto im_result = im_batch_data->im_result;
    im_data->CopyBatchData(im_result, &batch_spec, materialized_vertices,
                           computed_unmaterialized_vertices);
  }
  virtual void ReleaseBatch(size_t cur_exec_level, BatchData *im_batch_data,
                            BatchSpec batch_spec) {
    auto wctx = static_cast<LightWorkContext *>(wctx_);
    auto im_data = wctx->im_data;
    im_data->Release();
  }

  virtual bool NeedSearchNext(size_t cur_exec_level) {
    auto wctx = static_cast<LightWorkContext *>(wctx_);
    auto plan = wctx->plan;
    auto &exec_seq = plan->GetExecuteOperations();
    auto &materialized_vertices =
        plan->GetMaterializedVertices()[cur_exec_level];
    auto im_data = wctx->im_data;
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

  static void Count(LightWorkContext *wctx, size_t cur_exec_level) {
    auto plan = wctx->plan;
    auto &ans = *wctx->ans;
    auto &exec_seq = plan->GetExecuteOperations();
    bool allow_direct_count =
        plan->GetIntraPartitionPlanCompressLevel() ==
        LazyTraversalCompressLevel::COMPRESS_LEVEL_SPECIAL;
    QueryType query_type = plan->GetQuery()->GetQueryType();
    if (allow_direct_count && CanDirectCount(query_type)) {
      ans += LIGHTDirectCount(wctx, cur_exec_level);
    } else {
      ans += LIGHTCount(wctx, cur_exec_level);
    }
  }

  virtual void ExecuteBatch(size_t cur_exec_level, BatchSpec batch_spec) {
    auto wctx = static_cast<LightWorkContext *>(wctx_);
    auto task = static_cast<IntraPartTask *>(task_);
    auto d_partition_id = wctx->d_partition_id;
    auto context = wctx->context;
    auto plan = wctx->plan;
    auto im_data = wctx->im_data;
    auto gpu_profiler = wctx->gpu_profiler;
    auto &ans = *wctx->ans;

    auto &exec_seq = plan->GetExecuteOperations();
    auto &materialized_vertices =
        plan->GetMaterializedVertices()[cur_exec_level];
    auto &computed_unmaterialized_vertices =
        plan->GetComputedUnmaterializedVertices()[cur_exec_level];
    auto op = exec_seq[cur_exec_level].first;

    if (cur_exec_level == 0) {
      assert(op == MATERIALIZE);
      gpu_profiler->StartTimer("materialize_time", d_partition_id,
                               context->Stream());
      InitFirstLevel(wctx, task, exec_seq[cur_exec_level].second, &batch_spec);
      gpu_profiler->EndTimer("materialize_time", d_partition_id,
                             context->Stream());
      return;
    }

    switch (op) {
      case COMPUTE: {
        gpu_profiler->StartTimer("compute_time", d_partition_id,
                                 context->Stream());
        ComputeGeneral<false>(
            wctx, exec_seq[cur_exec_level].second,
            im_data->GetInstances()[materialized_vertices[0]]->GetSize());
        gpu_profiler->EndTimer("compute_time", d_partition_id,
                               context->Stream());

      } break;
      case FILTER_COMPUTE: {
        gpu_profiler->StartTimer("filter_compute_time", d_partition_id,
                                 context->Stream());
        FilterCompute(wctx, exec_seq[cur_exec_level].second);
        gpu_profiler->EndTimer("filter_compute_time", d_partition_id,
                               context->Stream());

      } break;
      case MATERIALIZE: {
        gpu_profiler->StartTimer("materialize_time", d_partition_id,
                                 context->Stream());
        Materialize(wctx, exec_seq[cur_exec_level].second,
                    materialized_vertices, computed_unmaterialized_vertices);

        gpu_profiler->EndTimer("materialize_time", d_partition_id,
                               context->Stream());

      } break;
      case COMPUTE_COUNT: {
        gpu_profiler->StartTimer("compute_count_time", d_partition_id,
                                 context->Stream());
        ans += ComputeCount(wctx, cur_exec_level);
        gpu_profiler->EndTimer("compute_count_time", d_partition_id,
                               context->Stream());
      } break;
      case COMPUTE_PATH_COUNT: {
        gpu_profiler->StartTimer("compute_path_count_time", d_partition_id,
                                 context->Stream());
        ComputePathCount(wctx, cur_exec_level);
        gpu_profiler->EndTimer("compute_path_count_time", d_partition_id,
                               context->Stream());
      } break;
      case COUNT: {
        gpu_profiler->StartTimer("count_time", d_partition_id,
                                 context->Stream());
        Count(wctx, cur_exec_level);
        gpu_profiler->EndTimer("count_time", d_partition_id, context->Stream());

      } break;
      default:
        assert(false);
        break;
    }
  }

  virtual void OrganizeBatch(size_t cur_exec_level,
                             BatchManager *batch_manager) {
    auto wctx = static_cast<LightWorkContext *>(wctx_);
    auto task = static_cast<IntraPartTask *>(task_);
    auto context = wctx->context;
    auto plan = wctx->plan;
    auto im_data = wctx->im_data;

    auto &exec_seq = plan->GetExecuteOperations();
    auto &materialized_vertices =
        plan->GetMaterializedVertices()[cur_exec_level];
    auto &computed_unmaterialized_vertices =
        plan->GetComputedUnmaterializedVertices()[cur_exec_level];

    if (cur_exec_level == 0) {
      size_t remaining_levels_num = exec_seq.size() - cur_exec_level;
      size_t parent_factor = sizeof(uintV);
      size_t temporary_parent_factor =
          std::ceil(1.0 * sizeof(size_t) * 3 /
                    remaining_levels_num);  // OrganizeBatch in next level
      parent_factor += temporary_parent_factor;
      batch_manager->OrganizeBatch(task->GetVertexCount(), parent_factor);
      return;
    }

    size_t path_num =
        im_data->GetInstances()[materialized_vertices[0]]->GetSize();
    auto op = exec_seq[cur_exec_level].first;

    DeviceArray<size_t> *children_count = NULL;
    size_t parent_factor = 0;
    size_t children_factor = 0;

    switch (op) {
      case COMPUTE: {
        BuildIntersectChildrenCount(wctx, exec_seq[cur_exec_level].second,
                                    path_num, children_count);

        EstimateComputeMemoryCost(exec_seq, cur_exec_level, parent_factor,
                                  children_factor);

      } break;
      case FILTER_COMPUTE: {
        BuildGatherChildrenCount(wctx, exec_seq[cur_exec_level].second,
                                 path_num, children_count);

        EstimateFilterComputeMemoryCost(exec_seq, cur_exec_level, parent_factor,
                                        children_factor);

      } break;
      case MATERIALIZE: {
        BuildGatherChildrenCount(wctx, exec_seq[cur_exec_level].second,
                                 path_num, children_count);

        EstimateMaterializeMemoryCost(
            exec_seq, materialized_vertices, computed_unmaterialized_vertices,
            cur_exec_level, parent_factor, children_factor);

      } break;
      case COMPUTE_COUNT: {
        // TODO: children_count can be avoided here
        children_count = new DeviceArray<size_t>(path_num, context);
        GpuUtils::Transform::Apply<ASSIGNMENT>(children_count->GetArray(),
                                               path_num, (size_t)1, context);
        EstimateComputeCountMemoryCost(exec_seq, cur_exec_level, parent_factor,
                                       children_factor);

      } break;
      case COMPUTE_PATH_COUNT: {
        BuildIntersectChildrenCount(wctx, exec_seq[cur_exec_level].second,
                                    path_num, children_count);
        EstimateComputePathCountMemoryCost(exec_seq, cur_exec_level,
                                           parent_factor, children_factor);
      } break;
      case COUNT: {
        EstimateCountMemoryCost(wctx, cur_exec_level, children_count,
                                parent_factor, children_factor);
      } break;
      default:
        assert(false);
        break;
    }
    batch_manager->OrganizeBatch(children_count, parent_factor, children_factor,
                                 children_count->GetSize(), context);
    delete children_count;
    children_count = NULL;
  }

  virtual void PrintProgress(size_t cur_exec_level, size_t batch_id,
                             size_t batch_num, BatchSpec batch_spec) {
#if defined(PROFILE)
    auto wctx = static_cast<LightWorkContext *>(wctx_);
    auto d_partition_id = wctx->d_partition_id;
    auto context = wctx->context;
    auto plan = wctx->plan;
    auto &ans = *wctx->ans;
    auto &exec_seq = plan->GetExecuteOperations();
    std::cout << "cur_exec_level=" << cur_exec_level << ",("
              << GetLazyTraversalOperationString(exec_seq[cur_exec_level].first)
              << "," << exec_seq[cur_exec_level].second
              << "),d_partition_id=" << d_partition_id
              << ",batch_id=" << batch_id << ",batch_num=" << batch_num
              << ", batch_spec=[" << batch_spec.GetBatchLeftEnd() << ","
              << batch_spec.GetBatchRightEnd() << "]"
              << ",ans=" << ans << ",available_size="
              << context->GetDeviceMemoryInfo()->GetAvailableMemorySizeMB()
              << "MB" << std::endl;
#endif
  }

  virtual void CollectCount(size_t cur_exec_level) {
#if defined(PROFILE)
    auto wctx = static_cast<LightWorkContext *>(wctx_);
    auto plan = wctx->plan;
    auto &exec_seq = plan->GetExecuteOperations();
    auto &materialized_vertices =
        plan->GetMaterializedVertices()[cur_exec_level];

    auto d_partition_id = wctx->d_partition_id;
    auto im_data = wctx->im_data;
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