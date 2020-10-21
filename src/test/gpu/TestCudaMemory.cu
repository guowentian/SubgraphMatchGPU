#include <cstdlib>
#include <iostream>
#include <vector>
#include "CudaContextManager.cuh"
#include "GPUTimer.cuh"
#include "Transform.cuh"

CudaContextManager *CudaContextManager::gCudaContextManager = NULL;

static float TestMemoryAllocate(int times) {
  const size_t dev_id = 0;
  CUDA_ERROR(cudaSetDevice(dev_id));
  CudaContext *context =
      CudaContextManager::GetCudaContextManager()->GetCudaContext(dev_id);

  const size_t small_memory_batch_size = 1024ULL * 1024 * 1024;
  const size_t large_memory_batch_size = 2ULL * 1024 * 1024 * 1024;
  GPUTimer timer;
  timer.StartTimer();
  for (int i = 0; i < times; ++i) {
    void *mem = NULL;
    mem = context->Malloc(small_memory_batch_size);
    context->Free(mem, small_memory_batch_size);

    mem = context->Malloc(large_memory_batch_size);
    context->Free(mem, large_memory_batch_size);
  }
  timer.EndTimer();
  return timer.GetElapsedMilliSeconds();
}

static float TestMemoryAllocateAndAccess(int times) {
  const size_t dev_id = 0;
  CUDA_ERROR(cudaSetDevice(dev_id));
  CudaContext *context =
      CudaContextManager::GetCudaContextManager()->GetCudaContext(dev_id);

  const size_t small_memory_batch_size = 1024ULL * 1024 * 1024;
  const size_t large_memory_batch_size = 2ULL * 1024 * 1024 * 1024;
  const size_t access_count = 100000000;
  GPUTimer timer;
  timer.StartTimer();
  for (int i = 0; i < times; ++i) {
    void *mem = NULL;
    char *data = NULL;
    mem = context->Malloc(small_memory_batch_size);
    data = (char *)mem;
    // cudaMemset(data, 0, sizeof(char) * small_memory_batch_size);
    GpuUtils::Transform::Transform([=] DEVICE(int index) { data[index] = 'a'; },
                                   access_count, context);
    context->Free(mem, small_memory_batch_size);

    mem = context->Malloc(large_memory_batch_size);
    data = (char *)mem;
    // cudaMemset(data, 0, sizeof(char) * large_memory_batch_size);
    GpuUtils::Transform::Transform([=] DEVICE(int index) { data[index] = 'a'; },
                                   access_count, context);
    context->Free(mem, large_memory_batch_size);
  }
  timer.EndTimer();
  return timer.GetElapsedMilliSeconds();
}

int main(int argc, char **argv) {
  int times = 100;
  if (argc == 2) {
    times = atoi(argv[1]);
  }
  const size_t gigabytes = 1024ULL * 1024 * 1024;
  std::vector<float> run_times;
  for (size_t i = 1; i <= 8; ++i) {
    CudaContextManager::CreateCudaContextManager(2, CNMEM_MANAGED);
    const size_t dev_id = 0;
    CUDA_ERROR(cudaSetDevice(dev_id));
    CudaContext *context =
        CudaContextManager::GetCudaContextManager()->GetCudaContext(dev_id);

    size_t consumed = gigabytes * i;
    void *mem = context->Malloc(consumed);
    // float t = TestMemoryAllocate(times);
    float t = TestMemoryAllocateAndAccess(times);
    run_times.push_back(t);
    context->Free(mem, consumed);

    CudaContextManager::FreeCudaContextManager();
  }

  std::cout << "times=" << times << std::endl;
  for (size_t i = 0; i < 8; ++i) {
    std::cout << i + 1 << "GB consumed, elapsed_time=" << run_times[i]
              << std::endl;
  }

  return 0;
}
