#include <gtest/gtest.h>
#include <thrust/sort.h>
#include "BatchManager.cuh"
#include "Compact.cuh"
#include "Copy.cuh"
#include "CudaContext.cuh"
#include "CudaContextManager.cuh"
#include "DevGraph.cuh"
#include "DevPlan.cuh"
#include "DeviceArray.cuh"
#include "GPUFilter.cuh"
#include "GPUUtil.cuh"
#include "Intersect.cuh"
#include "LoadBalance.cuh"
#include "SegmentReduce.cuh"
#include "TestGPUCommon.cuh"
#include "ThrustContext.cuh"
#include "Transform.cuh"

using namespace mgpu;

// global helper function
template <typename T>
T GetValue(T *val, cudaMemcpyKind kind = cudaMemcpyDeviceToHost) {
  assert(kind == cudaMemcpyDeviceToHost);
  T ret;
  CUDA_ERROR(cudaMemcpy(&ret, val, sizeof(T), kind));
  return ret;
}
template <typename T>
void SetValue(T *from, T *to, cudaMemcpyKind kind) {
  CUDA_ERROR(cudaMemcpy(from, to, sizeof(T), kind));
}
template <typename T>
T *AllocDevPtr() {
  T *ret = NULL;
  CUDA_ERROR(cudaMalloc(&ret, sizeof(T)));
  return ret;
}
template <typename T>
T *AllocDevPtr(T *h_ptr) {
  T *d_ptr = AllocDevPtr<T>();
  SetValue<T>(h_ptr, d_ptr, cudaMemcpyHostToDevice);
  return d_ptr;
}
template <typename T>
void FreePointer(T *val) {
  CUDA_ERROR(cudaFree(val));
}

// ===============================================
//
TEST(BasicUtils, NewDeviceArray) {
  const size_t d_partition_id = 0;
  CUDA_ERROR(cudaSetDevice(d_partition_id));
  CudaContext *context =
      CudaContextManager::GetCudaContextManager()->GetCudaContext(
          d_partition_id);

  DeviceArray<uintV> *arr = NULL;
  ReAllocate(arr, 10, context);

  ASSERT_EQ(arr->GetSize(), 10);

  delete arr;
  arr = NULL;
}

// =================================================

TEST(ParallelOperations, Compact) {
  const size_t d_partition_id = 0;
  CUDA_ERROR(cudaSetDevice(d_partition_id));
  CudaContext *context =
      CudaContextManager::GetCudaContextManager()->GetCudaContext(
          d_partition_id);

  const int N = 100;
  std::vector<int> h_input(N);
  bool *h_bitmaps = new bool[N];
  for (int i = 0; i < N; ++i) {
    h_input[i] = i;
    h_bitmaps[i] = i % 2;
  }
  std::vector<int> h_output;
  for (int i = 0; i < N; ++i) {
    if (i % 2) {
      h_output.push_back(i);
    }
  }

  DeviceArray<int> *d_input = new DeviceArray<int>(N, context);
  HToD(d_input->GetArray(), h_input.data(), N);
  DeviceArray<bool> *d_bitmaps = new DeviceArray<bool>(N, context);
  HToD(d_bitmaps->GetArray(), h_bitmaps, N);
  DeviceArray<int> *d_output = NULL;
  int output_count = 0;

  GpuUtils::Compact::Compact(d_input, N, d_bitmaps->GetArray(), d_output,
                             output_count, context);

  std::vector<int> actual_output(output_count);
  DToH(actual_output.data(), d_output->GetArray(), output_count);
  ASSERT_EQ(output_count, h_output.size());
  for (int i = 0; i < output_count; ++i) {
    ASSERT_EQ(actual_output[i], h_output[i]);
  }

  delete d_input;
  d_input = NULL;
  delete d_bitmaps;
  d_bitmaps = NULL;
  delete d_output;
  d_output = NULL;

  delete[] h_bitmaps;
  h_bitmaps = NULL;
}

TEST(ParallelOperations, Gather) {
  const size_t d_partition_id = 0;
  CUDA_ERROR(cudaSetDevice(d_partition_id));
  CudaContext *context =
      CudaContextManager::GetCudaContextManager()->GetCudaContext(
          d_partition_id);

  const int kInputCount = 20;
  const int kIndicesCount = 10;
  std::vector<int> h_input(kInputCount);
  std::vector<int> h_indices(kIndicesCount);
  for (int i = 0; i < kInputCount; ++i) {
    h_input.push_back(i);
    if (i < kIndicesCount) {
      h_indices.push_back(i);
    }
  }
  std::vector<int> exp_output(h_indices);

  DeviceArray<int> *d_indices = new DeviceArray<int>(kIndicesCount, context);
  DeviceArray<int> *d_input = new DeviceArray<int>(kInputCount, context);
  DeviceArray<int> *d_output = new DeviceArray<int>(kIndicesCount, context);
  HToD(d_indices->GetArray(), h_indices.data(), kIndicesCount);
  HToD(d_input->GetArray(), h_input.data(), kInputCount);
  GpuUtils::Copy::Gather(d_indices->GetArray(), kIndicesCount,
                         d_input->GetArray(), d_output->GetArray(), context);

  std::vector<int> actual_output(kIndicesCount);
  DToH(actual_output.data(), d_output->GetArray(), kIndicesCount);
  for (int i = 0; i < kIndicesCount; ++i) {
    ASSERT_EQ(actual_output[i], exp_output[i]);
  }
}

static void LoadBalanceTransformTestWrapper() {
  const size_t d_partition_id = 0;
  CUDA_ERROR(cudaSetDevice(d_partition_id));
  CudaContext *context =
      CudaContextManager::GetCudaContextManager()->GetCudaContext(
          d_partition_id);
  TrackPartitionedGraph *cpu_graph = Generate6Clique();
  DevGraph *dev_graph = new DevGraph(cpu_graph, context);

  size_t vertex_count = cpu_graph->GetVertexCount();
  size_t edge_count = cpu_graph->GetEdgeCount();
  DeviceArray<uintV> *output = new DeviceArray<uintV>(edge_count, context);
  uintV *output_array = output->GetArray();
  uintE *row_ptrs_array = dev_graph->GetRowPtrs()->GetArray();
  uintV *cols_array = dev_graph->GetCols()->GetArray();
  auto f = [=] DEVICE(int index, int seg, int rank) {
    output_array[index] = cols_array[row_ptrs_array[seg] + rank];
  };
  GpuUtils::LoadBalance::LBSTransform(f, edge_count, row_ptrs_array,
                                      vertex_count, context);

  std::vector<uintV> actual_output(edge_count);
  DToH(actual_output.data(), output_array, edge_count);
  for (size_t i = 0; i < edge_count; ++i) {
    ASSERT_EQ(actual_output[i], cpu_graph->GetCols()[i]);
  }

  delete output;
  output = NULL;
  delete cpu_graph;
  cpu_graph = NULL;
  delete dev_graph;
  dev_graph = NULL;
}
TEST(ParallelOperations, LoadBalanceTransform) {
  LoadBalanceTransformTestWrapper();
}

TEST(ParallelOperations, LoadBalanceSearch) {
  const size_t d_partition_id = 0;
  CUDA_ERROR(cudaSetDevice(d_partition_id));
  CudaContext *context =
      CudaContextManager::GetCudaContextManager()->GetCudaContext(
          d_partition_id);

  const int segments_num = 5;
  int total = 0;
  std::vector<int> h_segments(segments_num + 1);
  for (int i = 0; i < segments_num; ++i) {
    h_segments[i] = total;
    total += i + 1;
  }
  h_segments[segments_num] = total;
  std::vector<int> exp_output(total);
  for (int seg_id = 0; seg_id < segments_num; ++seg_id) {
    for (size_t j = h_segments[seg_id]; j < h_segments[seg_id + 1]; ++j) {
      exp_output[j] = seg_id;
    }
  }

  DeviceArray<int> *d_segments =
      new DeviceArray<int>(segments_num + 1, context);
  DeviceArray<int> *output = new DeviceArray<int>(total, context);
  HToD(d_segments->GetArray(), h_segments.data(), segments_num + 1);
  GpuUtils::LoadBalance::LoadBalanceSearch(
      total, d_segments->GetArray(), segments_num, output->GetArray(), context);

  Verify(output, exp_output.data(), total);

  delete d_segments;
  d_segments = NULL;
  delete output;
  output = NULL;
}

TEST(ParallelOperations, Scan) {
  const size_t d_partition_id = 0;
  CUDA_ERROR(cudaSetDevice(d_partition_id));
  CudaContext *context =
      CudaContextManager::GetCudaContextManager()->GetCudaContext(
          d_partition_id);

  const int N = 10;
  int sum = 0;
  std::vector<int> h_input(N);
  std::vector<int> h_output(N + 1);
  for (int i = 0; i < N; ++i) {
    h_input[i] = i;
    h_output[i] = sum;
    sum += i;
  }
  h_output[N] = sum;

  DeviceArray<int> *d_input = new DeviceArray<int>(N, context);
  HToD(d_input->GetArray(), h_input.data(), N);
  DeviceArray<int> *d_output = new DeviceArray<int>(N + 1, context);
  GpuUtils::Scan::ExclusiveSum(d_input->GetArray(), N, d_output->GetArray(),
                               d_output->GetArray() + N, context);

  std::vector<int> actual_output(N + 1);
  DToH(actual_output.data(), d_output->GetArray(), N + 1);
  for (int i = 0; i <= N; ++i) {
    ASSERT_EQ(actual_output[i], h_output[i]);
  }

  delete d_input;
  d_input = NULL;
  delete d_output;
  d_output = NULL;
}

static void SegmentReduceWrapper() {
  const size_t d_partition_id = 0;
  CUDA_ERROR(cudaSetDevice(d_partition_id));
  CudaContext *context =
      CudaContextManager::GetCudaContextManager()->GetCudaContext(
          d_partition_id);

  const int kWidth = 3;
  const int kSegmentsNum = 100;
  const int N = kWidth * kSegmentsNum;
  std::vector<int> h_segments(kSegmentsNum + 1);
  int prefix_sum = 0;
  for (int i = 0; i <= kSegmentsNum; ++i) {
    h_segments[i] = prefix_sum;
    prefix_sum += kWidth;
  }

  std::vector<int> h_values(N);
  for (int i = 0; i < kSegmentsNum; ++i) {
    for (int j = h_segments[i]; j < h_segments[i + 1]; ++j) {
      h_values[j] = i;
    }
  }
  DeviceArray<int> *d_values = new DeviceArray<int>(N, context);
  HToD(d_values->GetArray(), h_values.data(), N);
  int *d_values_array = d_values->GetArray();
  auto f = [=] DEVICE(int index) { return d_values_array[index]; };

  std::vector<int> exp_output(kSegmentsNum);
  for (int i = 0; i < kSegmentsNum; ++i) {
    exp_output[i] = kWidth * i;
  }

  DeviceArray<int> *d_segments =
      new DeviceArray<int>(kSegmentsNum + 1, context);
  HToD(d_segments->GetArray(), h_segments.data(), kSegmentsNum + 1);
  DeviceArray<int> *d_output = new DeviceArray<int>(kSegmentsNum, context);
  GpuUtils::SegReduce::TransformSegReduce(f, N, d_segments->GetArray(),
                                          kSegmentsNum, d_output->GetArray(), 0,
                                          context);

  std::vector<int> actual_output(kSegmentsNum);
  DToH(actual_output.data(), d_output->GetArray(), kSegmentsNum);
  for (int i = 0; i < kSegmentsNum; ++i) {
    ASSERT_EQ(actual_output[i], exp_output[i]);
  }

  delete d_values;
  d_values = NULL;
  delete d_output;
  d_output = NULL;
  delete d_segments;
  d_segments = NULL;
}
TEST(ParallelOperations, SegmentReduce) { SegmentReduceWrapper(); }

TEST(ParallelOperations, Transform) {
  const size_t d_partition_id = 0;
  CUDA_ERROR(cudaSetDevice(d_partition_id));
  CudaContext *context =
      CudaContextManager::GetCudaContextManager()->GetCudaContext(
          d_partition_id);

  const size_t N = 10;
  std::vector<int> h_input(N, 0);
  DeviceArray<int> *d_input = new DeviceArray<int>(N, context);
  HToD(d_input->GetArray(), h_input.data(), N);
  GpuUtils::Transform::Apply<ADD>(d_input->GetArray(), N, 10, context);

  std::vector<int> actual_output(N);
  DToH(actual_output.data(), d_input->GetArray(), N);
  for (size_t i = 0; i < N; ++i) {
    ASSERT_EQ(actual_output[i], 10);
  }

  delete d_input;
  d_input = NULL;
}

static void TransformLambdaWrapper() {
  const size_t d_partition_id = 0;
  CUDA_ERROR(cudaSetDevice(d_partition_id));
  CudaContext *context =
      CudaContextManager::GetCudaContextManager()->GetCudaContext(
          d_partition_id);

  const size_t N = 10;
  std::vector<int> h_input(N);
  for (size_t i = 0; i < N; ++i) {
    h_input[i] = i;
  }
  DeviceArray<int> *d_input = new DeviceArray<int>(N, context);
  DeviceArray<int> *d_output = new DeviceArray<int>(N, context);
  HToD(d_input->GetArray(), h_input.data(), N);
  int *d_input_array = d_input->GetArray();
  int *d_output_array = d_output->GetArray();

  GpuUtils::Transform::Transform(
      [=] DEVICE(int index) {
        d_output_array[index] = d_input_array[index] + index;
      },
      N, context);

  std::vector<int> actual_output(N);
  DToH(actual_output.data(), d_output->GetArray(), N);
  for (size_t i = 0; i < N; ++i) {
    ASSERT_EQ(actual_output[i], h_input[i] + i);
  }

  delete d_input;
  d_input = NULL;
  delete d_output;
  d_output = NULL;
}
TEST(ParallelOperations, TransformLambda) { TransformLambdaWrapper(); }

// ====================================================
TEST(Tools, BatchManagerTestOneLevel) {
  size_t dev_id = 0;
  CUDA_ERROR(cudaSetDevice(dev_id));
  CudaContext *context =
      CudaContextManager::GetCudaContextManager()->GetCudaContext(dev_id);

  size_t batch_size = 4;
  BatchManager *batch_manager = new BatchManager(context, batch_size);
  size_t parent_count = 3;
  batch_manager->OrganizeBatch(parent_count, sizeof(int));

  for (int i = 0; i < parent_count; ++i) {
    BatchSpec batch_spec = batch_manager->GetBatch(i);
    ASSERT_EQ(i, batch_spec.GetBatchLeftEnd());
    ASSERT_EQ(i, batch_spec.GetBatchRightEnd());
  }

  delete batch_manager;
  batch_manager = NULL;
}

// =================================================
TEST(BasicUtils, CacheCudaContextTest) {
  const size_t d_partition_id = 0;
  CUDA_ERROR(cudaSetDevice(d_partition_id));
  DeviceMemoryInfo *dev_mem =
      new DeviceMemoryInfo(d_partition_id, 1ULL * 1024 * 1024 * 1024 * 8);
  CacheCudaContext *context = new CacheCudaContext(dev_mem, 0);

  size_t before_size = context->GetDeviceMemoryInfo()->GetAvailableMemorySize();
  size_t cache_size = 10000000;
  context->MallocCache(cache_size);
  context->SetMallocFromCache(true);

  const int N = 10000;
  {
    mem_t<int> arr(N, *context);
    CUDA_ERROR(cudaMemset(arr.data(), 0, sizeof(int) * N));
  }

  {
    int *d_arr = (int *)context->Malloc(N * sizeof(int));
    CUDA_ERROR(cudaMemset(d_arr, 0, sizeof(int) * N));
  }

  context->SetMallocFromCache(false);
  context->FreeCache();
  size_t after_size = context->GetDeviceMemoryInfo()->GetAvailableMemorySize();
  ASSERT_EQ(before_size, after_size);

  delete context;
  context = NULL;
  delete dev_mem;
  dev_mem = NULL;
}

static void CnmemCudaContextTestWrapper(CudaContextType cuda_context_type) {
  CudaContextManager *manager = new CudaContextManager(1, cuda_context_type);
  const size_t d_partition_id = 0;
  CUDA_ERROR(cudaSetDevice(d_partition_id));
  CnmemCudaContext *context =
      static_cast<CnmemCudaContext *>(manager->GetCudaContext(d_partition_id));

  size_t before_size = context->GetDeviceMemoryInfo()->GetAvailableMemorySize();

  size_t large_base = 1024ULL * 1024 * 1024;
  size_t small_base = 1024;

  ///// test small allocation
  std::vector<void *> mem_ptrs;
  std::vector<size_t> mem_sizes;

  for (size_t i = 0; i < 10; ++i) {
    void *ptr = context->Malloc((i + 1) * small_base);
    mem_sizes.push_back((i + 1) * small_base);
    mem_ptrs.push_back(ptr);
  }
  for (size_t i = 0; i < 10; ++i) {
    if (i % 2 == 1) {
      context->Free(mem_ptrs[i], mem_sizes[i]);
    }
  }
  for (size_t i = 0; i < 10; ++i) {
    if (i % 2 == 0) {
      context->Free(mem_ptrs[i], mem_sizes[i]);
    }
  }

  // test large allocation
  mem_ptrs.resize(0);
  mem_sizes.resize(0);
  for (size_t i = 0; i < 10; ++i) {
    void *ptr = context->Malloc((i + 1) * large_base);
    mem_sizes.push_back((i + 1) * large_base);
    mem_ptrs.push_back(ptr);
  }
  for (size_t i = 0; i < 10; ++i) {
    if (i % 2 == 1) {
      context->Free(mem_ptrs[i], mem_sizes[i]);
    }
  }
  for (size_t i = 0; i < 10; ++i) {
    if (i % 2 == 0) {
      context->Free(mem_ptrs[i], mem_sizes[i]);
    }
  }

  size_t after_size = context->GetDeviceMemoryInfo()->GetAvailableMemorySize();
  ASSERT_EQ(before_size, after_size);

  delete manager;
  manager = NULL;
}
TEST(BasicUtils, CnmemCudaContextTest) {
  CnmemCudaContextTestWrapper(CudaContextType::CNMEM);
}
TEST(BasicUtils, CnmemManagedCudaContextTest) {
  CnmemCudaContextTestWrapper(CudaContextType::CNMEM_MANAGED);
}

CudaContextManager *CudaContextManager::gCudaContextManager = NULL;

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  CudaContextManager::CreateCudaContextManager(2, BASIC);
  return RUN_ALL_TESTS();
}
