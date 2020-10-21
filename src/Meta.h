#ifndef __META_H__
#define __META_H__

#include <cstddef>
#include <cstdint>
#include <limits>

typedef unsigned int uintV;        // vertex ids
typedef unsigned long long uintE;  // edge ids
typedef char uintP;                // partition ids
typedef int WeightType;

const static uintV kMaxuintV = std::numeric_limits<uintV>::max();
const static uintV kMinuintV = std::numeric_limits<uintV>::min();
const static uintE kMaxuintE = std::numeric_limits<uintE>::max();
const static uintE kMinuintE = std::numeric_limits<uintE>::min();
const static size_t kMaxsize_t = std::numeric_limits<size_t>::max();

// GPU
const static size_t THREADS_PER_BLOCK = 256;
const static size_t THREADS_PER_WARP = 32;
const static size_t WARPS_PER_BLOCK = THREADS_PER_BLOCK / THREADS_PER_WARP;
const static size_t MAX_BLOCKS_NUM = 96 * 8;
const static size_t MAX_THREADS_NUM = MAX_BLOCKS_NUM * THREADS_PER_BLOCK;

const static size_t kDeviceMemoryLimits[3] = {
    (size_t)(8.0 * 1024 * 1024 * 1024), (size_t)(8.0 * 1024 * 1024 * 1024),
    (size_t)(8.0 * 1024 * 1024 * 1024)};

const static size_t kMaxQueryVerticesNum = 6;
// sample
const size_t kRandomWalkSampleNum = 10240;
const double ALPHA = 0.1;
// O3 tuning parameters
const static double EXPAND_FACTOR = 3.0;
const static double PRUNE_FACTOR = 1.0 / 3;
const static unsigned int FULL_WARP_MASK = 0xffffffff;
// Reuse Traversal optimizer
const static double kAvgVertexDegree =
    12;  // average size of each adjacent list
const static double kRatioRandomAccessVersusSeqAccess =
    10;  // the ratio of random memory access versus sequential access in GPU
const static double kAvgSizeTreeNode = 12;  // the average size of children
                                            // nodes for a tree node, which is
                                            // set to be the same as kAvgDegree

enum Algo {
  // ============== CPU methods ===============
  // worst case optimal join
  CPU_WCOJ,
  // parallel VF2
  CPU_VF2,
  // basic BFS approach; for comparison with CPU_RBFS
  CPU_BFS,
  // reusable BFS on CPUs
  CPU_RBFS,

  // ============== GPU methods ===============
  // basic BFS approach on GPUs
  // "Fast subgraph matching on large graphs using graphics processors". 2015.
  GPU_GPSM,
  // reusable BFS based on GPU_GPSM
  GPU_GPSM_REUSE,
  // "GPU-Accelerated Subgraph Enumeration on Partitioned Graphs". 2020.
  GPU_LIGHT
};
enum ExecuteMode {
  // gpu for intra-partition instances, cpu for inter-partition instances
  HYBRID_CPU_GPU,
  // For small graphs that can fit the GPU memory.
  // GPU process all instances.
  IN_MEMORY_GPU,
  // External memory model.
  // Support large graphs that cannot fit the GPU memory.
  // GPU process all instances
  EXTERNAL_GPU,
};
// variant for HYBRID_CPU_GPU
enum Variant {
  // ======= GPU-CPU separate processing =====
  O0,  // CPU processes inter-partition workload without grouping
  O1,  // CPU processes inter-partition workload with grouping
  // ======= GPU-CPU interplay =======
  O2,  // GPU processes both intra and inter-partition workload while CPU load
       // subgraph from main memory to assist GPU in inter-partition search
       // ========= GPU-CPU coprocessing =========
  O3,  // O2 + CPU also participate in inter-partition workload with CPU_LIGHT
};

// different variants of GPU_LIGHT to search inter-partition instances in hybrid
// co-processing mode
enum GpuLightItpVariant {
  SINGLE_SEQUENCE,  // search each sequence one by one
  SHARED_EXECUTION  // search a group of search sequences altogether
};

enum GpuLightExtVariant {
  EXT_CACHE,  // by default, load the subgraph from main memory to GPU memory if
              // necessary
  EXT_UMA,  // do not load graph data manually but use the hardware feature UMA,
            // which serves for comparison
};

enum QueryType {
  Q0,   // TRIANGLE,
  Q1,   // SQUARE
  Q2,   // CHORDAL_SQUARE
  Q3,   // FOUR_CLIQUE,
  Q4,   // HOUSE
  Q5,   // QUAD_TRIANGLE
  Q6,   // NEAR5CLIQUE
  Q7,   // 5CLIQUE
  Q8,   // CHORDAL_ROOF
  Q9,   // THREE_TRIANGLE
  Q10,  // SOLAR_SQUARE
  Q11,  // 6CLIQUE
  Q12,  // NEAR_NEAR_5_CLIQUE
  Q13,  // TRIANGLE_IN_DIAMOND

  kQueryType,
  // one line with two vertices, for partitioner
  // only for preprocess partitioning, not for query pattern
  LINE
};

enum CondOperator { LESS_THAN, LARGER_THAN, NON_EQUAL, OPERATOR_NONE };
enum OperatorType { ADD, MULTIPLE, MAX, MIN, ASSIGNMENT, MINUS };

#endif
