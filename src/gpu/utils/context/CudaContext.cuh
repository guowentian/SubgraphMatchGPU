#ifndef __CUDA_CONTEXT_CUH__
#define __CUDA_CONTEXT_CUH__

#include <cassert>
#include <cstdio>
#if !defined(DISABLE_MGPU)
#include <moderngpu/context.hxx>
#endif
#include <vector>
#include "CudaContextUtils.h"
#include "GPUTimer.cuh"
#include "GPUUtil.cuh"
#include "Meta.h"

class CudaContextProfiler {
 public:
  CudaContextProfiler() { memory_operations_time_ = 0; }

  void StartTimer() {
#if defined(CUDA_CONTEXT_PROFILE)
    timer.StartTimer();
#endif
  }
  void EndTimer() {
#if defined(CUDA_CONTEXT_PROFILE)
    timer.EndTimer();
    memory_operations_time_ += timer.GetElapsedMilliSeconds();
#endif
  }
  double GetMemoryOperationsTime() const { return memory_operations_time_; }

 private:
  double memory_operations_time_;
  GPUTimer timer;
};

// A basic implementation of CudaContext.
// If DISABLE_MGPU is enabled, we inherit the memory operation API in
// standard_context_t so that the memory allocation in library calls of
// moderngpu can use the memory allocated in this CudaContext. The memory
// operations are basic: malloc and release device memory if the APIs are called
#if !defined(DISABLE_MGPU)
class CudaContext : public mgpu::standard_context_t {
 public:
  CudaContext(DeviceMemoryInfo *dev_mem, cudaStream_t stream)
      : mgpu::standard_context_t(false, stream), dev_mem_(dev_mem) {}

  ////////////////////////////////////
  // these API is specially for mgpu context to allocate memory
  virtual void *alloc(size_t size, mgpu::memory_space_t space) {
    if (space == mgpu::memory_space_device) {
      return Malloc(size);
    } else {
      return mgpu::standard_context_t::alloc(size, space);
    }
  }

  // we have modified the original interface to pass 'size '
  virtual void free(void *p, size_t size, mgpu::memory_space_t space) {
    if (space == mgpu::memory_space_device) {
      Free(p, size);
    } else {
      mgpu::standard_context_t::free(p, size, space);
    }
  }

  cudaStream_t Stream() { return stream(); }

#else  // if DISABLE_MGPU == true
class CudaContext {
 public:
  CudaContext(DeviceMemoryInfo *dev_mem, cudaStream_t stream)
      : dev_mem_(dev_mem), stream_(stream) {}

  cudaStream_t Stream() { return stream_; }

 protected:
  cudaStream_t stream_;
#endif

 public:
  ///// basic memory operation API //////
  virtual void *Malloc(size_t size) {
    cuda_context_profiler_.StartTimer();
    void *ret = SafeMalloc(size);
    // Memory statistics is updated after allocation because
    // the allocator needs to behave according to the current
    // available memory.
    dev_mem_->Consume(size);
    cuda_context_profiler_.EndTimer();
    return ret;
  }
  virtual void Free(void *p, size_t size) {
    cuda_context_profiler_.StartTimer();
    SafeFree(p);
    dev_mem_->Release(size);
    cuda_context_profiler_.EndTimer();
  }

  ///////// without tracking memory statistics /////
  // To support the case when the associated size of a pointer
  // cannot be abtained on Free, e.g., internal temporary memory
  // allocation in Thrust.
  // We should avoid use this API as much as possible.
  void *UnTrackMalloc(size_t size) {
    cuda_context_profiler_.StartTimer();
    void *ret = SafeMalloc(size);
    cuda_context_profiler_.EndTimer();
    return ret;
  }
  void UnTrackFree(void *p) {
    cuda_context_profiler_.StartTimer();
    SafeFree(p);
    cuda_context_profiler_.EndTimer();
  }

  /////  sync  /////
  void Synchronize() {
    cudaStream_t s = Stream();
    if (s) {
      CUDA_ERROR(cudaStreamSynchronize(s));
    } else {
      CUDA_ERROR(cudaDeviceSynchronize());
    }
  }

  ///////  info /////
  DeviceMemoryInfo *GetDeviceMemoryInfo() const { return dev_mem_; }

  void PrintProfileResult() const {
#if defined(CUDA_CONTEXT_PROFILE)
    std::cout << "dev_id=" << dev_mem_->GetDevId() << ",memory_operations_time="
              << cuda_context_profiler_.GetMemoryOperationsTime() << "ms"
              << std::endl;
#endif
  }

 protected:
  ////// malloc and free implementation /////
  // the inherited class is recommended to
  // override them to modify the implementation.
  // By default we use SafeMalloc and SafeFree.
  // DirectMalloc and DirectFree are only used for testing purpose.
  virtual void *DirectMalloc(size_t size) {
    void *ret = NULL;
    CUDA_ERROR(cudaMalloc(&ret, size));
    return ret;
  }

  virtual void DirectFree(void *p) { CUDA_ERROR(cudaFree(p)); }

  virtual void *SafeMalloc(size_t size) {
    if (dev_mem_->IsAvailable(size)) {
      return DirectMalloc(size);
    } else {
      fprintf(stderr, "Insufficient device memory\n");
      void *ret = NULL;
      // allocate from unified memory
      CUDA_ERROR(cudaMallocManaged(&ret, size));
      CUDA_ERROR(cudaMemPrefetchAsync(ret, size, dev_mem_->GetDevId()));
      return ret;
    }
  }

  virtual void SafeFree(void *p) { CUDA_ERROR(cudaFree(p)); }

 protected:
  DeviceMemoryInfo *dev_mem_;
  // The implementation of cuda_context_profiler_ is hidden by the flag
  // CUDA_CONTEXT_PROFILE. If such a flag is not specified, it will not cause
  // add extra behavior  for memory operations
  CudaContextProfiler cuda_context_profiler_;
};

// CacheCudaContext can mannually allocate a large portion of device memory
// and then keep malloc memory from that portion and release the whole portion
// of cache memory after the whole usage.
// This is useful to isolate the memory allocation overhead when testing the
// component, e.g., when testing performance of set intersection, and when
// process a batch during subgraph enumeration.
// CacheCudaContext is not needed as we have CnmemCudaContext.
// We keep it just because of some legacy code.
class CacheCudaContext : public CudaContext {
 public:
  CacheCudaContext(DeviceMemoryInfo *dev_mem, cudaStream_t stream)
      : CudaContext(dev_mem, stream) {
    malloc_from_cache_ = false;
  }
  ///////////////////////////////////////////////
  // general memory allocation using cuda API (expensive)
  virtual void *Malloc(size_t size) {
    cuda_context_profiler_.StartTimer();
    void *ret = NULL;
    if (!malloc_from_cache_) {
      ret = SafeMalloc(size);
      dev_mem_->Consume(size);
    } else {
      ret = MallocFromCache(size);
    }
    cuda_context_profiler_.EndTimer();
    return ret;
  }
  virtual void Free(void *p, size_t size) {
    cuda_context_profiler_.StartTimer();
    if (!malloc_from_cache_) {
      SafeFree(p);
      dev_mem_->Release(size);
    }
    cuda_context_profiler_.EndTimer();
  }

  // cache
  void SetMallocFromCache(bool f) { malloc_from_cache_ = f; }
  void MallocCache(size_t size) {
    assert(!malloc_from_cache_);
    void *base = Malloc(size);
    cache_alloc_.Init(base, size);
  }
  void FreeCache() {
    assert(!malloc_from_cache_);
    Free(cache_alloc_.GetBase(), cache_alloc_.GetSize());
    cache_alloc_.Reset();
  }

 protected:
  void *MallocFromCache(size_t size) { return cache_alloc_.Malloc(size); }

  NoFreeCacheAllocator cache_alloc_;
  // if malloc_from_cache_=true, keep allocating from cache_alloc_
  // without releasing memory
  bool malloc_from_cache_;
};

#include "cnmem.h"

// CnmemCudaContext assumes CNMEM_FLAGS_MANAGED is not used.
// The allocation from cnmem allocator may reach the buffer limit and return
// CNMEM_STATUS_OUT_OF_MEMORY. In that case, we mannually call cudaMallocManaged
// to allocate from the unified memory. We assume that case is seldom, so
// cudaMallocManaged and cudaFree the requested memory if needed.
class CnmemCudaContext : public CudaContext {
 public:
  CnmemCudaContext(DeviceMemoryInfo *dev_mem, cudaStream_t stream)
      : CudaContext(dev_mem, stream) {
    cnmem_device_ = new cnmemDevice_t();
    cnmem_device_->device = dev_mem_->GetDevId();
    cnmem_device_->size = dev_mem_->GetAvailableMemorySize();
    // In our use case, we have only one stream controls one device
    cnmem_device_->numStreams = 1;
#if !defined(DISABLE_MGPU)
    cnmem_device_->streams = &_stream;
#else
    cnmem_device_->streams = &stream_;
#endif
    // Do not specify the memory reserved for each stream, so that the memory is
    // allocated when needed
    cnmem_device_->streamSizes = NULL;
  }
  ~CnmemCudaContext() {
    delete cnmem_device_;
    cnmem_device_ = NULL;
  }

  cnmemDevice_t *GetCnmemDevice() const { return cnmem_device_; }

 protected:
  static void CnmemAssertSuccess(cnmemStatus_t status) {
    if (status != CNMEM_STATUS_SUCCESS) {
      std::cerr << cnmemGetErrorString(status) << std::endl;
      assert(false);
    }
  }

  virtual void *DirectMalloc(size_t size) {
    void *ret = NULL;
    cnmemStatus_t status = cnmemMalloc(&ret, size, Stream());
    CnmemAssertSuccess(status);
    return ret;
  }
  virtual void DirectFree(void *p) {
    cnmemStatus_t status = cnmemFree(p, Stream());
    CnmemAssertSuccess(status);
  }
  virtual void *SafeMalloc(size_t size) {
    void *ret = NULL;
    cnmemStatus_t status = cnmemMalloc(&ret, size, Stream());
    if (status == CNMEM_STATUS_OUT_OF_MEMORY) {
      fprintf(stderr, "Insufficient device memory\n");
      // allocate from unified memory
      CUDA_ERROR(cudaMallocManaged(&ret, size));
      CUDA_ERROR(cudaMemPrefetchAsync(ret, size, dev_mem_->GetDevId()));
    } else {
      CnmemAssertSuccess(status);
    }
    return ret;
  }
  virtual void SafeFree(void *p) {
    cnmemStatus_t status = cnmemFree(p, Stream());
    if (status == CNMEM_STATUS_INVALID_ARGUMENT) {
      // This pointer is not allocated from cnmem allocator
      CUDA_ERROR(cudaFree(p));
    } else {
      CnmemAssertSuccess(status);
    }
  }

 protected:
  cnmemDevice_t *cnmem_device_;
  size_t cnmem_stream_sizes_;
};

// CnmemManagedCudaContext assumes the flag is CNMEM_FLAGS_MANAGED and all the
// memory (including unified memory) is managed by the cnmem allocator. We do
// not need to worry about the case of large requested size.
class CnmemManagedCudaContext : public CnmemCudaContext {
 public:
  CnmemManagedCudaContext(DeviceMemoryInfo *dev_mem, cudaStream_t stream)
      : CnmemCudaContext(dev_mem, stream) {}
  ~CnmemManagedCudaContext() {}

 protected:
  virtual void *DirectMalloc(size_t size) {
    if (!dev_mem_->IsAvailable(size)) {
      fprintf(stderr, "Insufficient device memory\n");
    }
    void *ret = NULL;
    cnmemStatus_t status = cnmemMalloc(&ret, size, Stream());
    CnmemAssertSuccess(status);
    return ret;
  }
  virtual void DirectFree(void *p) {
    cnmemStatus_t status = cnmemFree(p, Stream());
    CnmemAssertSuccess(status);
  }

  virtual void *SafeMalloc(size_t size) { return DirectMalloc(size); }
  virtual void SafeFree(void *p) { DirectFree(p); }
};

#endif
