#ifndef __GPU_IMPL_COMMON_PROFILER_HELPER_CUH__
#define __GPU_IMPL_COMMON_PROFILER_HELPER_CUH__

#include "CountProfiler.h"
#include "GPUProfiler.cuh"

class ProfilerHelper {
 public:
  static void AddGPUProfilePhaseLight(GPUProfiler* gpu_profiler) {
    gpu_profiler->AddPhase("process_level_time");
    gpu_profiler->AddPhase("compute_time");
    gpu_profiler->AddPhase("compute_set_intersect_time");
    gpu_profiler->AddPhase("compute_count_time");
    gpu_profiler->AddPhase("compute_count_set_intersect_count_time");
    gpu_profiler->AddPhase("compute_path_count_time");
    gpu_profiler->AddPhase("filter_compute_time");
    gpu_profiler->AddPhase("filter_compute_lbs_expand_time");
    gpu_profiler->AddPhase("filter_compute_filter_time");
    gpu_profiler->AddPhase("materialize_time");
    gpu_profiler->AddPhase("materialize_lbs_expand_time");
    gpu_profiler->AddPhase("materialize_filter_time");
    gpu_profiler->AddPhase("materialize_copy_time");
    gpu_profiler->AddPhase("count_time");
  }
  static void AddGPUProfilePhaseGPSM(GPUProfiler* gpu_profiler) {
    gpu_profiler->AddPhase("total_process_time");
    gpu_profiler->AddPhase("process_level_time");

    gpu_profiler->AddPhase("check_constraints_time");
    gpu_profiler->AddPhase("materialize_time");
    gpu_profiler->AddPhase("check_constraints_first_step_time");
    gpu_profiler->AddPhase("check_constraints_second_step_time");
    gpu_profiler->AddPhase("join_phase_time");
    gpu_profiler->AddPhase("join_phase_kernel_time");
  }
  static void AddGPUProfilePhaseNemo(GPUProfiler* gpu_profiler) {
    gpu_profiler->AddPhase("total_process_time");
    gpu_profiler->AddPhase("process_level_time");
    gpu_profiler->AddPhase("organize_batch_time");
    gpu_profiler->AddPhase("generate_candidate_time");
    gpu_profiler->AddPhase("refine_candidate_time");
    gpu_profiler->AddPhase("condition_time");
    gpu_profiler->AddPhase("compact_time");
  }
  static void AddGPUProfilePhaseReuse(GPUProfiler* gpu_profiler) {
    gpu_profiler->AddPhase("reuse_find_cache_offsets");
  }
  static void AddGPUProfilePhaseEXT(GPUProfiler* gpu_profiler) {
    gpu_profiler->AddPhase("inspect_join_time");
    gpu_profiler->AddPhase("build_subgraph_time");
    gpu_profiler->AddPhase("wait_load_graph_time");
    gpu_profiler->AddPhase("prepare_load_graph_task_time");
    gpu_profiler->AddPhase("prepare_load_graph_task_copy_allocate_time");
    gpu_profiler->AddPhase("build_subgraph_row_ptrs_time");
    gpu_profiler->AddPhase("load_graph_htod_copy_edges_time");
  }

  static void AddCountProfilePhaseLight(CountProfiler* count_profiler) {
    count_profiler->AddPhase("generate_batch_count");
    count_profiler->AddPhase("compute_count");
    count_profiler->AddPhase("materialize_count");
  }
  static void AddCountProfilePhaseGPSM(CountProfiler* count_profiler) {
    count_profiler->AddPhase("generate_batch_count");
  }

  static void AddCountProfilePhaseReuse(CountProfiler* count_profiler) {
    count_profiler->AddPhase("intersect_count");
    count_profiler->AddPhase("reuse_count");
    count_profiler->AddPhase("hash_table_unique_eliminate_count");
  }
  static void AddCountProfilePhaseEXT(CountProfiler* count_profiler) {
    count_profiler->AddPhase("load_subgraph_pcie_send");
    count_profiler->AddPhase("load_subgraph_pcie_receive");
  }
};

#endif