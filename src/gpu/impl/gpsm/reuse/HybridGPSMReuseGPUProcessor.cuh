#ifndef __HYBRID_GPSM_REUSE_GPU_PROCESSOR_CUH__
#define __HYBRID_GPSM_REUSE_GPU_PROCESSOR_CUH__

#include "GPSMReuseCheckConstraints.cuh"
#include "GPSMReuseCommon.cuh"
#include "HybridGPSMGPUProcessor.cuh"
#include "ReuseTraversalPlan.h"

namespace GpsmReuse {
using namespace Gpsm;
class HybridGPSMReuseGPUProcessor : public HybridGPSMGPUProcessor {
 public:
  HybridGPSMReuseGPUProcessor(GpsmReuseWorkContext* wctx, Task* task)
      : HybridGPSMGPUProcessor(wctx, task) {}

  virtual void ExecuteBatch(size_t cur_level, BatchSpec batch_spec) {
    GpsmReuseWorkContext* wctx = static_cast<GpsmReuseWorkContext*>(wctx_);
    auto task = static_cast<IntraPartTask*>(task_);
    auto d_partition_id = wctx->d_partition_id;
    auto context = wctx->context;
    auto im_data = wctx->im_data;
    auto cache_data = wctx->cache_data;
    auto gpu_profiler = wctx->gpu_profiler;

    gpu_profiler->StartTimer("join_phase_time", d_partition_id,
                             context->Stream());
    if (cur_level == 0) {
      InitFirstLevel(wctx, cur_level, &batch_spec, task);
      auto& d_inst = im_data->GetInst();
      auto& d_inst_ptrs = cache_data->GetInstPtrs();
      d_inst_ptrs[cur_level] = d_inst[cur_level]->GetArray();
    } else {
      JoinPhase(wctx, cur_level, &batch_spec);
    }

    gpu_profiler->EndTimer("join_phase_time", d_partition_id,
                           context->Stream());
  }

  virtual void ReleaseBatch(size_t cur_level, BatchData* batch_data,
                            BatchSpec batch_spec) {
    GpsmReuseWorkContext* wctx = static_cast<GpsmReuseWorkContext*>(wctx_);
    auto im_data = wctx->im_data;
    auto cache_data = wctx->cache_data;
    auto plan = static_cast<ReuseTraversalPlan*>(wctx->plan);

    // As SetBatchData create a copy of the partial instances in the current
    // batch, we would release them, together with the instances for this
    // level
    // This release free the instances in the levels of [0,cur_level]
    im_data->Release();

    if (cur_level > 0) {
      // release cache_data
      if (cur_level < plan->GetVertexCount() - 1) {
        auto& d_inst_next_offsets = cache_data->GetInstNextOffsets();
        auto& d_inst_parents_indices = cache_data->GetInstParentsIndices();
        auto& d_inst_ptrs = cache_data->GetInstPtrs();
        auto& d_cache_next_offsets = cache_data->GetCacheNextOffsets();
        auto& d_cache_instances = cache_data->GetCacheInstances();
        ReleaseIfExists(d_inst_next_offsets[cur_level - 1]);
        ReleaseIfExists(d_inst_parents_indices[cur_level]);
        d_inst_ptrs[cur_level] = NULL;

        cache_data->SetLevelBatchoffset(cur_level - 1, 0, 0);
        if (plan->GetCacheRequired(cur_level)) {
          ReleaseIfExists(d_cache_next_offsets[cur_level - 1]);
          ReleaseIfExists(d_cache_instances[cur_level]);
        }
      }
    }
  }

  virtual void EstimateMemoryCost(uintV cur_level, size_t& parent_factor,
                                  size_t& children_factor) {
    auto wctx = static_cast<GpsmReuseWorkContext*>(wctx_);
    auto plan = static_cast<ReuseTraversalPlan*>(wctx->plan);
    // JoinPhase: reuse_valid, bool;
    // reuse_indices_vec, size_t * reuse_conn_meta.size();
    // overall_children_count, overall_children_offset, size_t;
    // intersect_children_count, intersect_children_offset,size_t
    auto& reuse_conn_meta = plan->GetLevelReuseIntersectPlan()[cur_level]
                                .GetReuseConnectivityMeta();
    parent_factor = sizeof(bool) + sizeof(size_t) * reuse_conn_meta.size() +
                    sizeof(size_t) * 4;

    // JoinPhase: overall_children
    // intersect_children
    // instances in the levels [0,cur_level]
    // + parents_indices
    children_factor =
        sizeof(uintV) * 2 + sizeof(uintV) * (cur_level + 1) + sizeof(uintV);

    // OrganizeBatch in next level: children_count+children_offset
    children_factor = std::max(children_factor, sizeof(size_t) * 2);
  }
};
}  // namespace GpsmReuse

#endif