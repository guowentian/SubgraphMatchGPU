#ifndef __HYBRID_GPSM_GPU_PROCESSOR_CUH__
#define __HYBRID_GPSM_GPU_PROCESSOR_CUH__

#include "CheckConstraintsCommon.cuh"
#include "Copy.cuh"
#include "CudaContext.cuh"
#include "DevTraversalPlan.cuh"
#include "GPSMCheckConstraints.cuh"
#include "GPSMCommon.cuh"
#include "GPUFilter.cuh"
#include "GPUTimer.cuh"
#include "HybridGPUProcessor.cuh"
#include "Scan.cuh"
#include "Task.h"
#include "Transform.cuh"

namespace Gpsm {
class HybridGPSMGPUProcessor : public HybridGPUProcessor {
 public:
  HybridGPSMGPUProcessor(GpsmWorkContext* wctx, Task* task)
      : HybridGPUProcessor(wctx, task) {}

  virtual bool IsProcessOver(size_t cur_level) {
    GpsmWorkContext* wctx = static_cast<GpsmWorkContext*>(wctx_);
    auto plan = wctx->plan;
    if (cur_level == plan->GetVertexCount()) {
      auto im_data = wctx->im_data;
      auto& ans = *wctx->ans;
      ans += im_data->GetInst()[cur_level - 1]->GetSize();
      return true;
    }
    return false;
  }

  virtual size_t GetLevelBatchSize(size_t cur_level) {
    GpsmWorkContext* wctx = static_cast<GpsmWorkContext*>(wctx_);
    auto context = wctx->context;
    auto plan = wctx->plan;
    size_t remaining_levels_num = plan->GetVertexCount() - cur_level;
    size_t P = context->GetDeviceMemoryInfo()->GetAvailableMemorySize() /
               remaining_levels_num;
    return BatchManager::GetSafeBatchSize(P);
  }

  virtual void BeforeBatchProcess(size_t cur_level, BatchData*& batch_data) {
    GpsmWorkContext* wctx = static_cast<GpsmWorkContext*>(wctx_);
    auto im_data = wctx->im_data;
    auto im_batch_data = new ImBatchData();
    im_batch_data->im_result = new ImResult(im_data->GetInst(), cur_level);
    batch_data = im_batch_data;
  }

  virtual void AfterBatchProcess(size_t cur_level, BatchData*& batch_data) {
    GpsmWorkContext* wctx = static_cast<GpsmWorkContext*>(wctx_);
    auto im_batch_data = static_cast<ImBatchData*>(batch_data);
    auto im_data = wctx->im_data;
    auto im_result = im_batch_data->im_result;
    // return back the partial instances, which would be released in the
    // previous frame
    im_result->SwapInstances(im_data->GetInst(), cur_level);
    delete im_result;
    im_result = NULL;
    delete im_batch_data;
    batch_data = NULL;
  }

  virtual void PrepareBatch(size_t cur_level, BatchData* batch_data,
                            BatchSpec batch_spec) {
    GpsmWorkContext* wctx = static_cast<GpsmWorkContext*>(wctx_);
    auto im_batch_data = static_cast<ImBatchData*>(batch_data);
    auto im_data = wctx->im_data;
    auto im_result = im_batch_data->im_result;
    // copy the partial instances that we need in this batch
    im_data->CopyBatchData(cur_level, im_result, &batch_spec);
  }

  virtual void ReleaseBatch(size_t cur_level, BatchData* batch_data,
                            BatchSpec batch_spec) {
    GpsmWorkContext* wctx = static_cast<GpsmWorkContext*>(wctx_);
    auto im_data = wctx->im_data;
    // As SetBatchData create a copy of the partial instances in the current
    // batch, we would release them, together with the instances for this
    // level
    // This release free the instances in the levels of [0,cur_level]
    im_data->Release();
  }

  virtual bool NeedSearchNext(size_t cur_level) {
    GpsmWorkContext* wctx = static_cast<GpsmWorkContext*>(wctx_);
    auto im_data = wctx->im_data;
    size_t path_num = im_data->GetInst()[cur_level]->GetSize();
    return path_num > 0;
  }

  virtual void ExecuteBatch(size_t cur_level, BatchSpec batch_spec) {
    GpsmWorkContext* wctx = static_cast<GpsmWorkContext*>(wctx_);
    IntraPartTask* task = static_cast<IntraPartTask*>(task_);
    auto d_partition_id = wctx->d_partition_id;
    auto context = wctx->context;
    auto gpu_profiler = wctx->gpu_profiler;
    if (cur_level == 0) {
      InitFirstLevel(wctx, cur_level, &batch_spec, task);

    } else {
      gpu_profiler->StartTimer("join_phase_time", d_partition_id,
                               context->Stream());

      JoinPhase(wctx, cur_level);

      // JoinPhaseOnePass(d_partition_id, context, cur_level, im_data,
      //                 im_data_holder, graph_partition, plan, dev_plan,
      //                 &batch_spec, gpu_profiler_, count_profiler_);
      gpu_profiler->EndTimer("join_phase_time", d_partition_id,
                             context->Stream());
    }
  }

  virtual void EstimateMemoryCost(uintV cur_level, size_t& parent_factor,
                                  size_t& children_factor) {
    // children_count+ children_offsets
    parent_factor = sizeof(size_t) * 2;
    // instances in the levels [0,cur_level]
    // + parents_indices
    children_factor = sizeof(uintV) * (cur_level + 1) + sizeof(size_t);
    // OrganizeBatch in next level: children_count+children_offset
    children_factor = std::max(children_factor, sizeof(size_t) * 2);
  }

  virtual void OrganizeBatch(size_t cur_level, BatchManager* batch_manager) {
    GpsmWorkContext* wctx = static_cast<GpsmWorkContext*>(wctx_);
    IntraPartTask* task = static_cast<IntraPartTask*>(task_);
    auto context = wctx->context;
    auto im_data = wctx->im_data;
    auto im_data_holder = wctx->im_data_holder;
    auto graph_partition = wctx->graph_partition;
    auto dev_plan = wctx->dev_plan;

    if (cur_level == 0) {
      size_t path_num = task->GetVertexCount();
      size_t parent_factor = sizeof(uintV);
      batch_manager->OrganizeBatch(path_num, parent_factor);
      return;
    }
    // prepare
    im_data_holder->GatherImData(im_data, context);
    auto d_seq_inst_data = im_data_holder->GetSeqInst()->GetArray();
    auto conn = dev_plan->GetBackwardConnectivity()->GetArray() + cur_level;
    uintE* row_ptrs = graph_partition->GetRowPtrs()->GetArray();

    auto& d_inst = im_data->GetInst();
    size_t parent_count = d_inst[cur_level - 1]->GetSize();

    // obtain children_countr
    DeviceArray<size_t>* children_count =
        new DeviceArray<size_t>(parent_count, context);
    size_t* children_count_data = children_count->GetArray();
    GpuUtils::Transform::Transform(
        [=] DEVICE(int index) {
          uintV M[kMaxQueryVerticesNum];
          for (size_t i = 0; i < conn->GetCount(); ++i) {
            uintV u = conn->Get(i);
            M[u] = d_seq_inst_data[u][index];
          }

          uintV pivot_level = ThreadChoosePivotLevel(*conn, M, row_ptrs);
          uintV pivot_vertex = M[pivot_level];
          children_count_data[index] =
              row_ptrs[pivot_vertex + 1] - row_ptrs[pivot_vertex];
        },
        parent_count, context);

    size_t parent_factor = 0;
    size_t children_factor = 0;
    EstimateMemoryCost(cur_level, parent_factor, children_factor);
    batch_manager->OrganizeBatch(children_count, parent_factor, children_factor,
                                 children_count->GetSize(), context);

    delete children_count;
    children_count = NULL;
  }

  virtual void PrintProgress(size_t cur_level, size_t batch_id,
                             size_t batch_num, BatchSpec batch_spec) {
#if defined(PROFILE)
    GpsmWorkContext* wctx = static_cast<GpsmWorkContext*>(wctx_);
    auto d_partition_id = wctx->d_partition_id;
    auto context = wctx->context;
    auto& ans = *wctx->ans;

    std::cout << "d_partition_id=" << d_partition_id
              << ",cur_level=" << cur_level << ",batch_id=" << batch_id
              << ",batch_num=" << batch_num << ", batch_spec=["
              << batch_spec.GetBatchLeftEnd() << ","
              << batch_spec.GetBatchRightEnd() << "]"
              << ",ans=" << ans << ",available_size="
              << context->GetDeviceMemoryInfo()->GetAvailableMemorySizeMB()
              << "MB" << std::endl;
#endif
  }
  virtual void CollectCount(size_t cur_level) {
    GpsmWorkContext* wctx = static_cast<GpsmWorkContext*>(wctx_);
    auto d_partition_id = wctx->d_partition_id;
    auto count_profiler = wctx->count_profiler;
    count_profiler->AddCount("generate_batch_count", d_partition_id, 1);
  }
};
}  // namespace Gpsm

#endif
