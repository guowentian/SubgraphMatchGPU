#ifndef __GPU_DEVICE_ARRAY_CUH__
#define __GPU_DEVICE_ARRAY_CUH__

#include <iostream>
#include <vector>
#include "CudaContext.cuh"
#include "GPUUtil.cuh"

// Some possible functionality for DeviceArray:
// 1. intrusive pointer s.t. the last owner for the device arrray is responsible
// to release the memory
// 2. assignment and copy constructor
// However, we want to restrict the service it provides and
// keep it as simple as possible for safety
template <typename T>
class DeviceArray {
 public:
  // The most commonly used constructor: DeviceArray is responsible to allocate
  // and release the memory
  DeviceArray(size_t size, CudaContext* context)
      : size_(size), context_(context) {
    alloc_ = true;
    array_ = (T*)context_->Malloc(sizeof(T) * size_);
  }
  // DeviceArray is only used as a place holder, or a pointer to a memory
  // region, so that it is not responsible to release the memory of array on the
  // deletion
  DeviceArray(T* array, size_t size) : array_(array), size_(size) {
    alloc_ = false;
    context_ = NULL;
  }

  // A general constructor, which should be used when necessary.
  // Possible use case: array is allocated somewhere outside, and
  // we want to release the memory of array automatically when DeviceArray is
  // deleted
  DeviceArray(T* array, size_t size, CudaContext* context, bool alloc)
      : array_(array), size_(size), context_(context), alloc_(alloc) {}

  ~DeviceArray() {
    if (alloc_) {
      context_->Free(array_, sizeof(T) * size_);
    }
    array_ = NULL;
  }

  // ============ getter ==============
  HOST_DEVICE size_t GetSize() const { return size_; }
  HOST_DEVICE T* GetArray() const { return array_; }
  DEVICE T Get(size_t idx) const { return array_[idx]; }
  CudaContext* GetContext() const { return context_; }

  // for debug
  void Print() const {
    std::cout << "size=" << size_ << ",";
    T* h_array = new T[size_];
    CUDA_ERROR(
        cudaMemcpy(h_array, array_, sizeof(T) * size_, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < size_; ++i) {
      std::cout << " " << h_array[i];
    }
    std::cout << std::endl;
    delete[] h_array;
    h_array = NULL;
  }

 private:
  T* array_;
  size_t size_;
  CudaContext* context_;
  bool alloc_;
};

// ================= some utilieis for DeviceArray =================

template <typename T>
using LayeredDeviceArray = std::vector<DeviceArray<T>*>;

template <typename T>
using TwoLayeredDeviceArray = std::vector<LayeredDeviceArray<T>>;

template <typename T>
void ReleaseIfExists(DeviceArray<T>*& arr) {
  if (arr) {
    delete arr;
    arr = NULL;
  }
}

template <typename T>
void ReAllocate(DeviceArray<T>*& arr, size_t count, CudaContext* context) {
  ReleaseIfExists(arr);
  arr = new DeviceArray<T>(count, context);
}

template <typename T>
void ReAllocate(DeviceArray<T>*& arr, T* init_array, size_t count) {
  ReleaseIfExists(arr);
  arr = new DeviceArray<T>(init_array, count);
}

// Given multiple DeviceArrays stored in a vector,
// build an array of DeviceArray in device memory
template <typename T>
void BuildTwoDimensionDeviceArray(DeviceArray<T*>*& to,
                                  const std::vector<DeviceArray<T>*>* from,
                                  CudaContext* context) {
  size_t dimension = from->size();
  std::vector<T*> h_arrays(dimension, NULL);
  for (size_t i = 0; i < from->size(); ++i) {
    h_arrays[i] = (from->at(i) == NULL) ? NULL : from->at(i)->GetArray();
  }
  ReAllocate(to, dimension, context);
  CUDA_ERROR(cudaMemcpy(to->GetArray(), h_arrays.data(), sizeof(T*) * dimension,
                        cudaMemcpyHostToDevice));
}

template <typename T>
void CopyTwoDimensionDeviceArray(DeviceArray<T*>* to,
                                 const std::vector<DeviceArray<T>*>& from) {
  size_t n = from.size();
  T** d_ptrs = new T*[n];
  for (size_t i = 0; i < n; ++i) {
    d_ptrs[i] = from[i] ? from[i]->GetArray() : NULL;
  }
  CUDA_ERROR(cudaMemcpy(to->GetArray(), d_ptrs, sizeof(T*) * n,
                        cudaMemcpyHostToDevice));
  delete[] d_ptrs;
  d_ptrs = NULL;
}

template <typename T>
void AsyncCopyTwoDimensionDeviceArray(DeviceArray<T*>* to,
                                      const std::vector<DeviceArray<T>*>& from,
                                      cudaStream_t stream) {
  size_t n = from.size();
  T** d_ptrs = new T*[n];
  for (size_t i = 0; i < n; ++i) {
    d_ptrs[i] = from[i] ? from[i]->GetArray() : NULL;
  }
  CUDA_ERROR(cudaMemcpyAsync(to->GetArray(), d_ptrs, sizeof(T*) * n,
                             cudaMemcpyHostToDevice, stream));
  delete[] d_ptrs;
  d_ptrs = NULL;
}

template <typename T>
static void PrintDeviceArray(DeviceArray<T>* d_array) {
  size_t n = d_array->GetSize();
  T* h_array = new T[n];
  DToH(h_array, d_array->GetArray(), n);
  for (size_t i = 0; i < n; ++i) {
    std::cout << i << ": " << h_array[i] << std::endl;
  }
  delete[] h_array;
  h_array = NULL;
}

#endif
