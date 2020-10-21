// For some unknown reasons, the headers for thrust should be included
// first. Otherwise, some definitions in thrust, e.g., parallel_for,
// would not be found. I suspect this may be because that the other
// libraries, e.g., cudpp and moderngpu also use thrust, there are some
// naming collisions and etc.
#include <thrust/sequence.h>

#include "EXTGPUPatternMatch.cuh"
#include "EXTLIGHTGPUComp.cuh"
#include "EXTUMALIGHTGPUComp.cuh"
#include "HybridCoprocessPatternMatch.cuh"
#include "HybridGPSMGPUComp.cuh"
#include "HybridGPSMReuseGPUComp.cuh"
#include "HybridInterplayPatternMatch.cuh"
#include "HybridLIGHTCPUComp.h"
#include "HybridLIGHTGPUComp.cuh"
#include "HybridLIGHTItpGPUComp.cuh"
#include "HybridLIGHTItpSSGPUComp.cuh"
#include "HybridPtGroupCPUComp.h"
#include "HybridSeparatePatternMatch.cuh"
#include "IMGPUPatternMatch.cuh"
#include "TraversalPlan.h"

#include <iostream>

static void InitGPUInstance(
    // command line parameters
    Algo algo, int thread_num, int dev_num, int partition_num,
    ExecuteMode execute_mode, Variant variant, bool materialize,
    std::string filename, bool directed, std::string partition_filename,
    QueryType query_id, GpuLightItpVariant gpu_light_itp_variant,
    GpuLightExtVariant gpu_light_ext_variant,
    LazyTraversalCompressLevel lazy_traversal_compress_level,
    bool enable_ordering,
    // output
    TrackPartitionedGraph *&cpu_graph, Query *&query, Plan *&plan,
    HybridGPUComponent *&gpu_comp, HybridCPUComponent *&cpu_comp,
    HybridGPUComponent *&itp_gpu_comp, PatternMatch *&pattern_match) {
  cpu_graph = new TrackPartitionedGraph(filename, directed, partition_filename,
                                        partition_num);
  query = new Query(query_id, enable_ordering);

  if (execute_mode == HYBRID_CPU_GPU) {
    // GPU component for intra-partition search
    switch (algo) {
      case GPU_GPSM_REUSE: {
        plan = new ReuseTraversalPlan(query, partition_num, dev_num,
                                      execute_mode, variant);
        gpu_comp = new GpsmReuse::HybridGpsmReusePipelineGPUComponent(
            (ReuseTraversalPlan *)plan, cpu_graph, materialize, thread_num);
      } break;
      case GPU_LIGHT: {
        plan =
            new LazyTraversalPlan(query, partition_num, dev_num, execute_mode,
                                  variant, lazy_traversal_compress_level);
        gpu_comp = new Light::HybridLIGHTPipelineGPUComponent(
            (LazyTraversalPlan *)plan, cpu_graph, materialize, thread_num);
      } break;
      default:
        assert(false);
        break;
    }

    switch (variant) {
      case O0:
      case O1: {
        // the inter-partition workload is handled by CPU component
        cpu_comp = new HybridPartialGroupCPUComponent(
            (TraversalPlan *)plan, cpu_graph, materialize, thread_num);

        pattern_match = new HybridSeparatePatternMatch(
            plan, cpu_graph, gpu_comp, cpu_comp, itp_gpu_comp, thread_num);
      } break;
      case O2:
      case O3: {
        // GPU component can process the inter-partition workload
        switch (algo) {
          case GPU_LIGHT: {
            switch (gpu_light_itp_variant) {
              case SINGLE_SEQUENCE: {
                itp_gpu_comp =
                    new Light::HybridLIGHTItpSingleSequenceGPUComponent(
                        (LazyTraversalPlan *)plan, cpu_graph, materialize,
                        thread_num);
              } break;
              case SHARED_EXECUTION: {
                itp_gpu_comp = new Light::HybridLIGHTItpPipelineGPUComponent(
                    (LazyTraversalPlan *)plan, cpu_graph, materialize,
                    thread_num);
              } break;
              default:
                assert(false);
                break;
            }
          } break;
          default:
            assert(false);
            break;
        }

        if (variant == O2) {
          // CPU component does not participate in inter-partition workload
          pattern_match = new HybridInterplayPatternMatch(
              plan, cpu_graph, gpu_comp, cpu_comp, itp_gpu_comp, thread_num);
        } else {
          // CPU component can also process inter-partition workload
          assert(algo == GPU_LIGHT);
          cpu_comp = new HybridLIGHTCPUComponent(
              (LazyTraversalPlan *)plan, cpu_graph, materialize, thread_num);

          pattern_match = new HybridCoprocessPatternMatch(
              plan, cpu_graph, gpu_comp, cpu_comp, itp_gpu_comp, thread_num);
        }
      } break;
      default:
        assert(false);
        break;
    }

  } else if (execute_mode == IN_MEMORY_GPU) {
    switch (algo) {
      case GPU_GPSM: {
        plan = new TraversalPlan(query, partition_num, dev_num, execute_mode,
                                 variant);
        gpu_comp = new Gpsm::HybridGpsmPipelineGPUComponent(
            (TraversalPlan *)plan, cpu_graph, materialize, thread_num);
      } break;
      case GPU_GPSM_REUSE: {
        plan = new ReuseTraversalPlan(query, partition_num, dev_num,
                                      execute_mode, variant);
        gpu_comp = new GpsmReuse::HybridGpsmReusePipelineGPUComponent(
            (ReuseTraversalPlan *)plan, cpu_graph, materialize, thread_num);
      } break;
      case GPU_LIGHT: {
        plan =
            new LazyTraversalPlan(query, partition_num, dev_num, execute_mode,
                                  variant, lazy_traversal_compress_level);
        gpu_comp = new Light::HybridLIGHTPipelineGPUComponent(
            (LazyTraversalPlan *)plan, cpu_graph, materialize, thread_num);
      } break;
      default:
        break;
    }

    pattern_match = new IMGPUPatternMatch(plan, cpu_graph, gpu_comp, cpu_comp,
                                          itp_gpu_comp, thread_num);

  } else if (execute_mode == EXTERNAL_GPU) {
    switch ((Algo)algo) {
      case GPU_LIGHT: {
        plan =
            new LazyTraversalPlan(query, partition_num, dev_num, execute_mode,
                                  variant, lazy_traversal_compress_level);
        switch (gpu_light_ext_variant) {
          case EXT_CACHE:
            gpu_comp = new Light::EXTLIGHTPipelineGPUComponent(
                (LazyTraversalPlan *)plan, cpu_graph, materialize, thread_num);
            break;
          case EXT_UMA:
            gpu_comp = new Light::EXTUMALIGHTPipelineGPUComponent(
                (LazyTraversalPlan *)plan, cpu_graph, materialize, thread_num);
            break;
          default:
            break;
        }
      } break;
      default:
        break;
    }

    pattern_match = new EXTGPUPatternMatch(plan, cpu_graph, gpu_comp, cpu_comp,
                                           itp_gpu_comp, thread_num);
  } else {
    assert(false);
  }
}

static void ReleaseGPUInstance(TrackPartitionedGraph *&cpu_graph, Query *&query,
                               Plan *&plan, HybridGPUComponent *&gpu_comp,
                               HybridCPUComponent *&cpu_comp,
                               HybridGPUComponent *&itp_gpu_comp,
                               PatternMatch *&pattern_match) {
  delete pattern_match;
  pattern_match = NULL;
  if (itp_gpu_comp) {
    delete itp_gpu_comp;
    itp_gpu_comp = NULL;
  }
  delete cpu_comp;
  cpu_comp = NULL;
  delete gpu_comp;
  gpu_comp = NULL;
  delete plan;
  plan = NULL;
  delete query;
  query = NULL;
  delete cpu_graph;
  cpu_graph = NULL;
}

static void PrintParameters(
    Algo algo, int thread_num, int dev_num, int partition_num,
    ExecuteMode execute_mode, Variant variant, bool materialize,
    std::string filename, bool directed, std::string partition_filename,
    QueryType query_type, GpuLightItpVariant gpu_light_itp_variant,
    GpuLightExtVariant gpu_light_ext_variant,
    LazyTraversalCompressLevel lazy_traversal_compress_level,
    bool enable_ordering, const size_t *dev_mem_limits,
    CudaContextType cuda_context_type) {
  std::cout << "filename=" << filename << ",directed=" << directed
            << ",algo=" << algo << ",query_type=" << query_type << std::endl;
  std::cout << ",partition_filename=" << partition_filename
            << ",partition_num=" << partition_num
            << ",thread_num=" << thread_num << ",execute_mode=" << execute_mode
            << ",variant=" << variant << ",materialize=" << materialize
            << ",gpu_light_itp_variant=" << gpu_light_itp_variant
            << ",lazy_traversal_compress_level="
            << lazy_traversal_compress_level
            << ",enable_ordering=" << enable_ordering
            << ",gpu_light_ext_variant=" << gpu_light_ext_variant << std::endl;
  std::cout << "device memory: ";
  for (size_t dev_id = 0; dev_id < dev_num; ++dev_id) {
    std::cout << " " << dev_mem_limits[dev_id] / 1024.0 / 1024.0 / 1024.0
              << "G";
  }
  std::cout << std::endl;
  std::cout << "cuda_context_type=" << cuda_context_type << std::endl;
}

static void PrintHelperMsg() {
  std::cout << "./patternmatchcpu -f FILENAME -d IS_DIRECTRED -a ALGORITHM "
               "-q QUERY -e PARTITION_FILENAME -p PARTITION_NUM "
               "-t THREAD_NUM -v DEVICE_NUM -m EXECUTE_MODE -o VARIANT [ -l "
               "MATERIALIZE -r ENABLE_ORDERING -c "
               "LAZY_TRAVERSAL_COMPRESS_LEVEL -x GPU_LIGHT_ITP_VARIANT -y "
               "GPU_LIGHT_EXT_VARIANT]"
            << std::endl;
  std::cout << "ALGORITHM: "
            << ",GPU_GPSM=" << GPU_GPSM << ",GPU_GPSM_REUSE=" << GPU_GPSM_REUSE
            << ",GPU_LIGHT=" << GPU_LIGHT << std::endl;
  std::cout << "QUERY: "
            << "Q0 (TRIANGLE) " << Q0 << ", Q1 (square) " << Q1
            << ", Q2 (chordal square) " << Q2 << ", Q3 (4 clique) " << Q3
            << ", Q4 (house) " << Q4 << ", Q5 (quad triangle) " << Q5
            << ", Q6 (near5clique) " << Q6 << ", Q7 (5 clique) " << Q7
            << ", Q8 (chordal roof) " << Q8 << ", Q9 (three triangle) " << Q9
            << ", Q10 (solar square) " << Q10 << ", Q11 (6 clique) " << Q11
            << std::endl;
  std::cout << "MODE: "
            << "HYBRID_CPU_GPU=" << HYBRID_CPU_GPU
            << ", IN_MEMORY_GPU=" << IN_MEMORY_GPU
            << ", EXTERNAL_GPU=" << EXTERNAL_GPU << std::endl;
  std::cout << "variant: "
            << "O0(SEPARATE WITHOUT SHARING)=" << O0
            << ", O1(SEPARATE WITH SHARING)=" << O1 << ", O2(INTERPLAY)=" << O2
            << ", O3(COPROCESSING)=" << O3 << std::endl;
  std::cout << "LAZY_TRAVERSAL_COMPRESS_LEVEL: COMPRESS_LEVEL_MATERIALIZE="
            << COMPRESS_LEVEL_MATERIALIZE << ", COMPRESS_LEVEL_NON_MATERIALIZE="
            << COMPRESS_LEVEL_NON_MATERIALIZE
            << ", COMPRESS_LEVEL_NON_MATERIALIZE_OPT="
            << COMPRESS_LEVEL_NON_MATERIALIZE_OPT
            << ", COMPRESS_LEVEL_SPECIAL(DEFAULT)=" << COMPRESS_LEVEL_SPECIAL
            << std::endl;
  std::cout << "GPU_LIGHT_ITP_VARIANT: "
            << "SINGLE_SEQUENCE=" << SINGLE_SEQUENCE
            << ", SHARED_EXECUTION(DEFAULT)=" << SHARED_EXECUTION << std::endl;
  std::cout << "GPU_LIGHT_EXT_VARIANT: EXT_CACHE(DEFAULT)="
            << GpuLightExtVariant::EXT_CACHE
            << ", EXT_UMA=" << GpuLightExtVariant::EXT_UMA << std::endl;
  ;
  exit(-1);
}