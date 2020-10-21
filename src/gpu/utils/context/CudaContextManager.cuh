#ifndef __GPU_UTILS_CUDA_CONTEXT_MANAGER_CUH__
#define __GPU_UTILS_CUDA_CONTEXT_MANAGER_CUH__

#include <iostream>
#include "CudaContext.cuh"
#include "cnmem.h"

enum CudaContextType {
  // malloc and free device memory on demand
  BASIC,
  // BASIC + can manually allocate a large portion of memory for usage
  CACHE,
  CNMEM,
  CNMEM_MANAGED,
};

class CudaContextManager {
 public:
  CudaContextManager(size_t dev_num, CudaContextType cuda_context_type)
      : dev_num_(dev_num), cuda_context_type_(cuda_context_type) {
    dev_mems_.resize(dev_num_);
    contexts_.resize(dev_num_);
    streams_.resize(dev_num_);
    cnmem_devices_ = NULL;
    for (size_t dev_id = 0; dev_id < dev_num_; ++dev_id) {
      dev_mems_[dev_id] =
          new DeviceMemoryInfo(dev_id, kDeviceMemoryLimits[dev_id], false);
      CUDA_ERROR(cudaSetDevice(dev_id));
      CUDA_ERROR(cudaStreamCreate(&streams_[dev_id]));
      switch (cuda_context_type_) {
        case BASIC: {
          contexts_[dev_id] =
              new CudaContext(dev_mems_[dev_id], streams_[dev_id]);
          break;
        }
        case CACHE: {
          contexts_[dev_id] =
              new CacheCudaContext(dev_mems_[dev_id], streams_[dev_id]);
          break;
        }
        case CNMEM: {
          contexts_[dev_id] =
              new CnmemCudaContext(dev_mems_[dev_id], streams_[dev_id]);
          break;
        }
        case CNMEM_MANAGED: {
          contexts_[dev_id] =
              new CnmemManagedCudaContext(dev_mems_[dev_id], streams_[dev_id]);
          break;
        }
        default:
          break;
      }
    }
    if (cuda_context_type_ == CNMEM || cuda_context_type_ == CNMEM_MANAGED) {
      // Optional flags to create cnmem allocators:
      // CNMEM_FLAGS_DEFAULT: allow to grow beyond the buffer limit and use
      // cudaMalloc.
      // CNMEM_FLAGS_CANNOT_GROW: cannot allow to grow beyond the
      // buffer limit and use cudaMalloc
      // CNMEM_FLAGS_CANNOT_GROW |CNMEM_FLAGS_MANAGED: cannot allow to grow
      // beyond the buffer limit and use cudaMallocManaged (thus this call can
      // always succeed)
      // CNMEM_FLAGS_MANAGED: allows to grow beyond buffer limit and always use
      // cudaMallocManaged, so in this case we will not need to worry about the
      // insufficient device memory as the cnmem allocator takes care of all the
      // cases
      cnmemStatus_t status = CNMEM_STATUS_SUCCESS;

      switch (cuda_context_type_) {
        case CNMEM: {
          cnmem_devices_ = new cnmemDevice_t[dev_num_];
          for (size_t dev_id = 0; dev_id < dev_num_; ++dev_id) {
            CnmemCudaContext *ctx =
                static_cast<CnmemCudaContext *>(contexts_[dev_id]);
            memcpy(cnmem_devices_ + dev_id, ctx->GetCnmemDevice(),
                   sizeof(cnmemDevice_t));
          }
          status = cnmemInit(dev_num_, cnmem_devices_,
                             cnmemManagerFlags_t::CNMEM_FLAGS_CANNOT_GROW);
          break;
        }
        case CNMEM_MANAGED: {
          cnmem_devices_ = new cnmemDevice_t[dev_num_];
          for (size_t dev_id = 0; dev_id < dev_num_; ++dev_id) {
            CnmemManagedCudaContext *ctx =
                static_cast<CnmemManagedCudaContext *>(contexts_[dev_id]);
            memcpy(cnmem_devices_ + dev_id, ctx->GetCnmemDevice(),
                   sizeof(cnmemDevice_t));
          }
          status = cnmemInit(dev_num_, cnmem_devices_,
                             cnmemManagerFlags_t::CNMEM_FLAGS_MANAGED);

          break;
        }
        default:
          break;
      }
      if (status != CNMEM_STATUS_SUCCESS) {
        std::cerr << cnmemGetErrorString(status);
      }
    }
  }
  ~CudaContextManager() {
    if (cuda_context_type_ == CNMEM) {
      cnmemStatus_t status = cnmemFinalize();
      if (status != CNMEM_STATUS_SUCCESS) {
        std::cerr << cnmemGetErrorString(status);
      }
      delete[] cnmem_devices_;
      cnmem_devices_ = NULL;
    }
    for (size_t dev_id = 0; dev_id < dev_num_; ++dev_id) {
      CUDA_ERROR(cudaSetDevice(dev_id));
      CUDA_ERROR(cudaStreamDestroy(streams_[dev_id]));
      switch (cuda_context_type_) {
        case BASIC: {
          CudaContext *ctx = contexts_[dev_id];
          delete ctx;
          break;
        }
        case CACHE: {
          CacheCudaContext *ctx =
              static_cast<CacheCudaContext *>(contexts_[dev_id]);
          delete ctx;
          break;
        }
        case CNMEM: {
          CnmemCudaContext *ctx =
              static_cast<CnmemCudaContext *>(contexts_[dev_id]);
          delete ctx;
          break;
        }
        case CNMEM_MANAGED: {
          CnmemManagedCudaContext *ctx =
              static_cast<CnmemManagedCudaContext *>(contexts_[dev_id]);
          delete ctx;
          break;
        }
        default:
          break;
      }
      contexts_[dev_id] = NULL;
      delete dev_mems_[dev_id];
      dev_mems_[dev_id] = NULL;
    }
  }
  CudaContext *GetCudaContext(size_t dev_id) const { return contexts_[dev_id]; }

  static CudaContextManager *GetCudaContextManager() {
    return gCudaContextManager;
  }
  static void CreateCudaContextManager(size_t dev_num,
                                       CudaContextType cuda_context_type) {
    gCudaContextManager = new CudaContextManager(dev_num, cuda_context_type);
  }
  static void FreeCudaContextManager() {
    delete gCudaContextManager;
    gCudaContextManager = NULL;
  }

  // adapt to existing implementation
  size_t GetAvailableMemorySize(size_t dev_id) const {
    return contexts_[dev_id]->GetDeviceMemoryInfo()->GetAvailableMemorySize();
  }
  size_t GetAvailableMemorySizeMB(size_t dev_id) const {
    return contexts_[dev_id]->GetDeviceMemoryInfo()->GetAvailableMemorySizeMB();
  }
  bool IsAvailable(size_t consume, size_t dev_id) const {
    return contexts_[dev_id]->GetDeviceMemoryInfo()->IsAvailable(consume);
  }

  void ConsumeMemory(size_t size, size_t dev_id) {
    assert(IsAvailable(size, dev_id));
    contexts_[dev_id]->GetDeviceMemoryInfo()->Consume(size);
  }
  void ReleaseMemory(size_t size, size_t dev_id) {
    contexts_[dev_id]->GetDeviceMemoryInfo()->Release(size);
  }
  size_t GetMemoryUsedSize(size_t dev_id) const {
    return contexts_[dev_id]->GetDeviceMemoryInfo()->GetMemoryUsedSize();
  }

 protected:
  CudaContextType cuda_context_type_;
  size_t dev_num_;
  std::vector<DeviceMemoryInfo *> dev_mems_;
  std::vector<CudaContext *> contexts_;
  std::vector<cudaStream_t> streams_;
  cnmemDevice_t *cnmem_devices_;  // for cnmem

  static CudaContextManager *gCudaContextManager;
};

#endif
