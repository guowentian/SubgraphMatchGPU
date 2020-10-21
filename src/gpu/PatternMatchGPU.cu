#include "Meta.h"

#include <cuda_profiler_api.h>
#include "CommandLine.h"
#include "CudaContext.cuh"
#include "CudaContextManager.cuh"
#include "PatternMatchGPUInstance.cuh"

CudaContextManager *CudaContextManager::gCudaContextManager = NULL;

int main(int argc, char *argv[]) {
  if (argc == 1) {
    PrintHelperMsg();
  }

  CommandLine cmd(argc, argv);
  std::string filename =
      cmd.GetOptionValue("-f", "../../data/com-dblp.ungraph.txt");
  int directed = cmd.GetOptionIntValue("-d", 1);
  int algo = cmd.GetOptionIntValue("-a", CPU_WCOJ);
  int query_type = cmd.GetOptionIntValue("-q", Q0);
  std::string partition_filename = cmd.GetOptionValue("-e", "");
  int partition_num = cmd.GetOptionIntValue("-p", 1);
  int thread_num = cmd.GetOptionIntValue("-t", 1);
  int dev_num = cmd.GetOptionIntValue("-v", 1);
  ExecuteMode execute_mode =
      (ExecuteMode)cmd.GetOptionIntValue("-m", HYBRID_CPU_GPU);
  Variant variant = (Variant)cmd.GetOptionIntValue("-o", O1);
  bool materialize = cmd.GetOptionIntValue("-l", false);
  GpuLightItpVariant gpu_light_itp_variant =
      (GpuLightItpVariant)cmd.GetOptionIntValue("-x", SHARED_EXECUTION);
  LazyTraversalCompressLevel lazy_traversal_compress_level =
      (LazyTraversalCompressLevel)cmd.GetOptionIntValue(
          "-c", LazyTraversalCompressLevel::COMPRESS_LEVEL_SPECIAL);
  bool enable_ordering = cmd.GetOptionIntValue("-r", true);
  GpuLightExtVariant gpu_light_ext_variant =
      (GpuLightExtVariant)cmd.GetOptionIntValue("-y", EXT_CACHE);
  // CudaContextType cuda_context_type = BASIC;
  // CudaContextType cuda_context_type = CNMEM;
  CudaContextType cuda_context_type = CNMEM_MANAGED;
  CudaContextManager::CreateCudaContextManager(dev_num, cuda_context_type);
  // const size_t main_memory_size = 1ULL * 1024 * 1024 * 1024 * 64;

  PrintParameters((Algo)algo, thread_num, dev_num, partition_num, execute_mode,
                  variant, materialize, filename, directed, partition_filename,
                  (QueryType)query_type, gpu_light_itp_variant,
                  gpu_light_ext_variant, lazy_traversal_compress_level,
                  enable_ordering, kDeviceMemoryLimits, cuda_context_type);

  TrackPartitionedGraph *cpu_graph = NULL;
  Query *query = NULL;
  Plan *plan = NULL;
  HybridGPUComponent *gpu_comp = NULL;
  HybridCPUComponent *cpu_comp = NULL;
  HybridGPUComponent *itp_gpu_comp = NULL;
  PatternMatch *pattern_match = NULL;

  InitGPUInstance((Algo)algo, thread_num, dev_num, partition_num, execute_mode,
                  variant, materialize, filename, directed, partition_filename,
                  (QueryType)query_type, gpu_light_itp_variant,
                  gpu_light_ext_variant, lazy_traversal_compress_level,
                  enable_ordering, cpu_graph, query, plan, gpu_comp, cpu_comp,
                  itp_gpu_comp, pattern_match);

#if defined(NVPROFILE)
  cudaProfilerStart();
#endif
  pattern_match->Execute();
#if defined(NVPROFILE)
  cudaProfilerStop();
#endif

  for (int i = 0; i < dev_num; ++i) {
    CudaContextManager::GetCudaContextManager()
        ->GetCudaContext(i)
        ->PrintProfileResult();
  }

  ReleaseGPUInstance(cpu_graph, query, plan, gpu_comp, cpu_comp, itp_gpu_comp,
                     pattern_match);

  return 0;
}
