#include <thrust/sequence.h>

#include "CudaContextManager.cuh"
#include "PatternMatchGPUInstance.cuh"

#include <gtest/gtest.h>

const long long ans[2][kQueryType] = {
    {2224385LL, 55107655LL, 105043837LL, 16713192LL, 16703493573LL,
     1590858494251LL, 8058169897LL, 262663639LL, 8101412570LL, 16406438768LL,
     3982336960LL, 4221802226LL, 7922128863ULL},
    {3056386LL, 468774021LL, 251755062LL, 4986965LL, 53073844144LL,
     1823152976463LL, 2943152115LL, 7211947LL, 5582737206LL, 20624052752LL,
     753156161LL, 8443803LL, 628664532ULL}};

const std::string DATASET_DIR = "../../../data";
const std::string DBLP_FILENAME =
    DATASET_DIR + "/" + "com-dblp.ungraph.txt.bin";
const std::string YOUTUBE_FILENAME =
    DATASET_DIR + "/" + "com-youtube.ungraph.txt.bin";

static size_t GetFileId(const std::string filename) {
  if (filename == DBLP_FILENAME) {
    return 0;
  } else if (filename == YOUTUBE_FILENAME) {
    return 1;
  } else {
    assert(false);
  }
  return (size_t)-1;
}

static void RunGPUInstance(Algo algo, int thread_num, int dev_num,
                           int partition_num, ExecuteMode execute_mode,
                           Variant variant, bool materialize,
                           std::string filename, bool directed,
                           std::string partition_filename, size_t query_id,
                           const long long query_ans,
                           GpuLightItpVariant gpu_light_itp_variant,
                           GpuLightExtVariant gpu_light_ext_variant,
                           LazyTraversalCompressLevel compress_level) {
  TrackPartitionedGraph *cpu_graph = NULL;
  Query *query = NULL;
  Plan *plan = NULL;
  HybridGPUComponent *gpu_comp = NULL;
  HybridCPUComponent *cpu_comp = NULL;
  HybridGPUComponent *itp_gpu_comp = NULL;
  PatternMatch *pattern_match = NULL;

  InitGPUInstance(algo, thread_num, dev_num, partition_num, execute_mode,
                  variant, materialize, filename, directed, partition_filename,
                  (QueryType)query_id, gpu_light_itp_variant,
                  gpu_light_ext_variant, compress_level, true, cpu_graph, query,
                  plan, gpu_comp, cpu_comp, itp_gpu_comp, pattern_match);

  pattern_match->Execute();
  ASSERT_EQ(pattern_match->GetTotalMatchCount(), query_ans);

  ReleaseGPUInstance(cpu_graph, query, plan, gpu_comp, cpu_comp, itp_gpu_comp,
                     pattern_match);
}

/// ================ entry ==============
static void RunGPUInstance(
    const std::string filename, Algo algo_type, int thread_num, int dev_num,
    int partition_num, ExecuteMode execute_mode, Variant variant,
    const size_t *queries, const size_t queries_num,
    GpuLightItpVariant gpu_light_itp_variant = SHARED_EXECUTION,
    GpuLightExtVariant gpu_light_ext_variant = EXT_CACHE,
    LazyTraversalCompressLevel compress_level = COMPRESS_LEVEL_SPECIAL) {
  bool materialize = false;
  bool directed = false;
  size_t file_id = GetFileId(filename);

  const std::string partition_filename_prefix = ".veweight.uniform.part.";
  std::string partition_filename =
      filename + partition_filename_prefix + std::to_string(partition_num);
  // if (execute_mode != HYBRID_CPU_GPU) {
  //  partition_filename = "";
  //  partition_num = 1;
  //}

  for (size_t idx = 0; idx < queries_num; ++idx) {
    size_t query_id = queries[idx];
    RunGPUInstance(algo_type, thread_num, dev_num, partition_num, execute_mode,
                   variant, materialize, filename, directed, partition_filename,
                   query_id, ans[file_id][query_id], gpu_light_itp_variant,
                   gpu_light_ext_variant, compress_level);
  }
}
static void RunGPUOnlyInstance(
    const std::string filename, Algo algo_type, int thread_num, int dev_num,
    ExecuteMode execute_mode, const size_t *queries, const size_t queries_num,
    LazyTraversalCompressLevel compress_level = COMPRESS_LEVEL_SPECIAL) {
  assert(execute_mode == IN_MEMORY_GPU || execute_mode == EXTERNAL_GPU);
  RunGPUInstance(filename, algo_type, thread_num, dev_num, 1, execute_mode, O0,
                 queries, queries_num, SHARED_EXECUTION, EXT_CACHE,
                 compress_level);
}

//// ===================  query ===================
static const size_t kFastQueries[] = {0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11};
static const size_t kFastQueriesNum = 11;
static const size_t kFullQueries[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
static const size_t kFullQueriesNum = 12;

// static const size_t kReuseQueries[] = {0, 2, 3, 5, 6, 7, 9, 10, 11};
// static const size_t kReuseQueriesNum = 9;
static const size_t kReuseFastQueries[] = {0, 2, 3, 6, 7, 9, 10, 11};
static const size_t kReuseFastQueriesNum = 8;

//// ================ overall tests =================

TEST(OverallTest, DISABLED_InMemoryGPUTest) {
  std::string filenames[2] = {DBLP_FILENAME, YOUTUBE_FILENAME};
  Algo algorithms[1] = {GPU_GPSM};
  const int thread_num = 2;
  const int dev_num = 1;
  ExecuteMode execute_mode = IN_MEMORY_GPU;
  for (size_t file_id = 0; file_id < 2; ++file_id) {
    for (size_t algo_id = 0; algo_id < 1; ++algo_id) {
      std::cout << "----- algo=" << algorithms[algo_id] << "------"
                << std::endl;
      RunGPUOnlyInstance(filenames[file_id], algorithms[algo_id], thread_num,
                         dev_num, execute_mode, kFastQueries, kFastQueriesNum);
    }
  }
  Algo reuse_algorithms[1] = {GPU_GPSM_REUSE};

  for (size_t file_id = 0; file_id < 2; ++file_id) {
    for (size_t algo_id = 0; algo_id < 1; ++algo_id) {
      std::cout << "----- algo=" << reuse_algorithms[algo_id] << "------"
                << std::endl;
      RunGPUOnlyInstance(filenames[file_id], reuse_algorithms[algo_id],
                         thread_num, dev_num, execute_mode, kReuseFastQueries,
                         kReuseFastQueriesNum);
    }
  }
}

//// ============= HYBRID_CPU_GPU test =================
// LIGHT
// HYBRID_CPU_GPU+O1
TEST(HybridTest, DISABLED_LightDblpO1) {
  RunGPUInstance(DBLP_FILENAME, GPU_LIGHT, 32, 1, 4, HYBRID_CPU_GPU, O1,
                 kFastQueries, kFastQueriesNum);
}
TEST(HybridTest, DISABLED_LightYoutubeO1) {
  RunGPUInstance(YOUTUBE_FILENAME, GPU_LIGHT, 32, 1, 4, HYBRID_CPU_GPU, O1,
                 kFastQueries, kFastQueriesNum);
}
// LIGHT
// HYBRID_CPU_GPU+O2, SHARED_EXECUTION
TEST(HybridTest, LightDblpO2SharedExecution) {
  RunGPUInstance(DBLP_FILENAME, GPU_LIGHT, 1, 1, 4, HYBRID_CPU_GPU, O2,
                 kFastQueries, kFastQueriesNum, SHARED_EXECUTION);
}
TEST(HybridTest, LightYoutubeO2SharedExecution) {
  RunGPUInstance(YOUTUBE_FILENAME, GPU_LIGHT, 1, 1, 4, HYBRID_CPU_GPU, O2,
                 kFastQueries, kFastQueriesNum, SHARED_EXECUTION);
}
// LIGHT
// HYBRID_CPU_GPU + O2, SINGLE_SEQUENCE
TEST(HybridTest, LightDblpO2SS) {
  RunGPUInstance(DBLP_FILENAME, GPU_LIGHT, 1, 1, 4, HYBRID_CPU_GPU, O2,
                 kFastQueries, kFastQueriesNum, SINGLE_SEQUENCE);
}
TEST(HybridTest, LightYoutubeO2SS) {
  RunGPUInstance(YOUTUBE_FILENAME, GPU_LIGHT, 1, 1, 4, HYBRID_CPU_GPU, O2,
                 kFastQueries, kFastQueriesNum, SINGLE_SEQUENCE);
}
// LIGHT
// HYBRID_CPU_GPU + O2, SHARED_EXECUTION, different LazyTraversalCompressLevel
TEST(HybridTest, LightDblpO2SharedExecutionCompressLevel) {
  LazyTraversalCompressLevel compress_levels[3] = {
      COMPRESS_LEVEL_NON_MATERIALIZE_OPT, COMPRESS_LEVEL_NON_MATERIALIZE,
      COMPRESS_LEVEL_MATERIALIZE};
  for (size_t i = 0; i < 3; ++i) {
    RunGPUInstance(DBLP_FILENAME, GPU_LIGHT, 10, 1, 4, HYBRID_CPU_GPU, O2,
                   kFastQueries, kFastQueriesNum, SHARED_EXECUTION, EXT_CACHE,
                   compress_levels[i]);
  }
}
TEST(HybridTest, LightYoutubeO2SharedExecutionCompressLevel) {
  LazyTraversalCompressLevel compress_levels[3] = {
      COMPRESS_LEVEL_NON_MATERIALIZE_OPT, COMPRESS_LEVEL_NON_MATERIALIZE,
      COMPRESS_LEVEL_MATERIALIZE};
  for (size_t i = 0; i < 3; ++i) {
    RunGPUInstance(YOUTUBE_FILENAME, GPU_LIGHT, 10, 1, 4, HYBRID_CPU_GPU, O2,
                   kFastQueries, kFastQueriesNum, SHARED_EXECUTION, EXT_CACHE,
                   compress_levels[i]);
  }
}

TEST(HybridTest, DISABLED_SpecificLightYoutubeO2SS) {
  const size_t queries[] = {8};
  RunGPUInstance(YOUTUBE_FILENAME, GPU_LIGHT, 1, 1, 4, HYBRID_CPU_GPU, O2,
                 queries, 1, SINGLE_SEQUENCE);
}

// LIGHT
// HYBRID_CPU_GPU + O3
TEST(HybridTest, LightDblpO3) {
  RunGPUInstance(DBLP_FILENAME, GPU_LIGHT, 10, 1, 4, HYBRID_CPU_GPU, O3,
                 kFastQueries, kFastQueriesNum, SHARED_EXECUTION);
}
TEST(HybridTest, LightYoutubeO3) {
  RunGPUInstance(YOUTUBE_FILENAME, GPU_LIGHT, 10, 1, 4, HYBRID_CPU_GPU, O3,
                 kFastQueries, kFastQueriesNum, SHARED_EXECUTION);
}
// LIGHT
// HYBRID_CPU_GPU + O3, SHARED_EXECUTION, different LazyTraversalCompressLevel
TEST(HybridTest, LightDblpO3SharedExecutionCompressLevel) {
  LazyTraversalCompressLevel compress_levels[3] = {
      COMPRESS_LEVEL_NON_MATERIALIZE_OPT, COMPRESS_LEVEL_NON_MATERIALIZE,
      COMPRESS_LEVEL_MATERIALIZE};
  for (size_t i = 0; i < 3; ++i) {
    RunGPUInstance(DBLP_FILENAME, GPU_LIGHT, 10, 1, 4, HYBRID_CPU_GPU, O3,
                   kFastQueries, kFastQueriesNum, SHARED_EXECUTION, EXT_CACHE,
                   compress_levels[i]);
  }
}
TEST(HybridTest, LightYoutubeO3SharedExecutionCompressLevel) {
  LazyTraversalCompressLevel compress_levels[3] = {
      COMPRESS_LEVEL_NON_MATERIALIZE_OPT, COMPRESS_LEVEL_NON_MATERIALIZE,
      COMPRESS_LEVEL_MATERIALIZE};
  for (size_t i = 0; i < 3; ++i) {
    RunGPUInstance(YOUTUBE_FILENAME, GPU_LIGHT, 10, 1, 4, HYBRID_CPU_GPU, O3,
                   kFastQueries, kFastQueriesNum, SHARED_EXECUTION, EXT_CACHE,
                   compress_levels[i]);
  }
}

///// ============= IN MEMORY GPU mode test ============
// GPU_GPSM
TEST(PatternMatchGPUTest, GpsmDblpTestMode1) {
  RunGPUOnlyInstance(DBLP_FILENAME, GPU_GPSM, 10, 1, IN_MEMORY_GPU,
                     kFastQueries, kFastQueriesNum);
}
TEST(PatternMatchGPUTest, GpsmYoutubeTestMode1) {
  RunGPUOnlyInstance(YOUTUBE_FILENAME, GPU_GPSM, 10, 1, IN_MEMORY_GPU,
                     kFastQueries, kFastQueriesNum);
}

// GPU_LIGHT
TEST(PatternMatchGPUTest, LightDblpTestMode1) {
  RunGPUOnlyInstance(DBLP_FILENAME, GPU_LIGHT, 10, 1, IN_MEMORY_GPU,
                     kFullQueries, kFullQueriesNum);
}
TEST(PatternMatchGPUTest, LightYoutubeTestMode1) {
  RunGPUOnlyInstance(YOUTUBE_FILENAME, GPU_LIGHT, 10, 1, IN_MEMORY_GPU,
                     kFullQueries, kFullQueriesNum);
}

///// =============== EXTERNAL_GPU  mode =================
// GPU_LIGHT
TEST(ExternalGPUTest, LightDblpTestMode1) {
  RunGPUOnlyInstance(DBLP_FILENAME, GPU_LIGHT, 10, 2, EXTERNAL_GPU,
                     kFullQueries, kFullQueriesNum);
}
TEST(ExternalGPUTest, LightYoutubeTestMode1) {
  RunGPUOnlyInstance(YOUTUBE_FILENAME, GPU_LIGHT, 10, 2, EXTERNAL_GPU,
                     kFullQueries, kFullQueriesNum);
}
// GPU_LIGHT, EXT_UMA
TEST(ExternalGPUTest, LightDblpTestUMA) {
  RunGPUInstance(DBLP_FILENAME, GPU_LIGHT, 10, 2, 1, EXTERNAL_GPU, O0,
                 kFullQueries, kFullQueriesNum, SHARED_EXECUTION, EXT_UMA);
}
TEST(ExternalGPUTest, LightYoutubeTestUMA) {
  RunGPUInstance(YOUTUBE_FILENAME, GPU_LIGHT, 10, 2, 1, EXTERNAL_GPU, O0,
                 kFullQueries, kFullQueriesNum, SHARED_EXECUTION, EXT_UMA);
}

//// =============   reuse queries ====================
TEST(PatternMatchGPUTest, GpsmReuseDblpTestMode1) {
  RunGPUOnlyInstance(DBLP_FILENAME, GPU_GPSM_REUSE, 10, 1, IN_MEMORY_GPU,
                     kReuseFastQueries, kReuseFastQueriesNum);
}
TEST(PatternMatchGPUTest, GpsmReuseYoutubeTestMode1) {
  RunGPUOnlyInstance(YOUTUBE_FILENAME, GPU_GPSM_REUSE, 10, 1, IN_MEMORY_GPU,
                     kReuseFastQueries, kReuseFastQueriesNum);
}

CudaContextManager *CudaContextManager::gCudaContextManager = NULL;

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  CudaContextManager::CreateCudaContextManager(2,
                                               CudaContextType::CNMEM_MANAGED);
  return RUN_ALL_TESTS();
}
