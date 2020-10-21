#ifndef __HYBRID_GPU_COMPONENT_COMMON_WORK_CONTEXT_CUH__
#define __HYBRID_GPU_COMPONENT_COMMON_WORK_CONTEXT_CUH__

#include "CPUGraph.h"
#include "CountProfiler.h"
#include "CudaContext.cuh"
#include "DevGraphPartition.cuh"
#include "GPUProfiler.cuh"

struct WorkContext {
  size_t d_partition_id;
  size_t thread_num;
  TrackPartitionedGraph* cpu_relation;
  long long* ans;

  CudaContext* context;
  DevGraphPartition* graph_partition;

  GPUProfiler* gpu_profiler;
  CountProfiler* count_profiler;

  WorkContext() {
    d_partition_id = 0;
    ans = NULL;
    context = NULL;
    graph_partition = NULL;
    gpu_profiler = NULL;
    count_profiler = NULL;
  }

  void Set(size_t d_partition_id, size_t thread_num,
           TrackPartitionedGraph* cpu_relation, long long* ans,
           CudaContext* context, DevGraphPartition* graph_partition,
           GPUProfiler* gpu_profiler, CountProfiler* count_profiler) {
    this->d_partition_id = d_partition_id;
    this->thread_num = thread_num;
    this->cpu_relation = cpu_relation;
    this->ans = ans;
    this->context = context;
    this->graph_partition = graph_partition;
    this->gpu_profiler = gpu_profiler;
    this->count_profiler = count_profiler;
  }
};

#endif