#ifndef __HYBRID_LIGHT_INTERPLAY_PIPELINE_GPU_COMPONENT_COMMON_CUH__
#define __HYBRID_LIGHT_INTERPLAY_PIPELINE_GPU_COMPONENT_COMMON_CUH__

#include "LIGHTCommon.cuh"

namespace Light {

struct LightItpWorkContext : LightWorkContext {
  size_t dfs_group_id;
  size_t dfs_id;
  std::vector<std::vector<DevLazyTraversalPlan*>>* dfs_dev_plan;

  LightItpWorkContext() : LightWorkContext() {
    dfs_group_id = 0;
    dfs_id = 0;
    dfs_dev_plan = NULL;
  }
  void Set(size_t _d_partition_id, size_t _thread_num,
           TrackPartitionedGraph* _cpu_relation, long long* _ans,
           CudaContext* _context, DevGraphPartition* _graph_partition,
           GPUProfiler* _gpu_profiler, CountProfiler* _count_profiler,
           LazyTraversalPlan* _plan, ImData* _im_data,
           ImDataDevHolder* _im_data_holder, DevLazyTraversalPlan* _dev_plan,
           GraphDevTracker* _graph_dev_tracker, size_t _dfs_group_id,
           size_t _dfs_id,
           std::vector<std::vector<DevLazyTraversalPlan*>>* _dfs_dev_plan) {
    LightWorkContext::Set(_d_partition_id, _thread_num, _cpu_relation, _ans,
                          _context, _graph_partition, _gpu_profiler,
                          _count_profiler, _plan, _im_data, _im_data_holder,
                          _dev_plan, _graph_dev_tracker);
    this->dfs_group_id = _dfs_group_id;
    this->dfs_id = _dfs_id;
    this->dfs_dev_plan = _dfs_dev_plan;
  }
};
}  // namespace Light

#endif