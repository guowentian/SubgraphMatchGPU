#ifndef __GPU_UTILS_CUDA_CONTEXT_UTILS_CUH__
#define __GPU_UTILS_CUDA_CONTEXT_UTILS_CUH__

#include <cassert>
#include <stack>

class DeviceMemoryInfo {
 public:
  DeviceMemoryInfo(size_t dev_id, size_t memory_limit, bool sync = false)
      : dev_id_(dev_id),
        memory_limit_(memory_limit),
        sync_(sync),
        memory_used_(0) {}

  size_t GetAvailableMemorySize() const {
    return memory_used_ <= memory_limit_ ? memory_limit_ - memory_used_ : 0;
  }

  size_t GetAvailableMemorySizeMB() const {
    return GetAvailableMemorySize() / 1000.0 / 1000.0;
  }

  bool IsAvailable(size_t consume) const {
    return GetAvailableMemorySize() >= consume;
  }
  size_t GetMemoryUsedSize() const { return memory_used_; }

  void Release(size_t size) {
    assert(memory_used_ >= size);
    memory_used_ -= size;
  }

  // Allow memory_used_ > memory_limit_ to support SafeMalloc from CudaContext
  void Consume(size_t size) { memory_used_ += size; }

  // memory size check and allocation in an atomic operation
  bool TryConsume(size_t size) {
    if (IsAvailable(size)) {
      Consume(size);
      return true;
    }
    return false;
  }

  size_t GetMemoryLimit() const { return memory_limit_; }
  size_t GetDevId() const { return dev_id_; }

 private:
  const size_t dev_id_;
  const size_t memory_limit_;
  // when sync_ = true, multiple streams may share the same device
  // so lock is requried to access the device memory info
  // TODO: the case for sync_ = true
  const bool sync_;
  size_t memory_used_;
};

// thread-local cache manager
class CacheAllocator {
 public:
  CacheAllocator() { Reset(); }
  CacheAllocator(void *b, size_t size) { Init(b, size); }

  void Init(void *b, size_t size) {
    base_ = b;
    size_ = size;
    used_ = 0;
  }

  void Reset() { Init(NULL, 0); }

  bool IsAvailable(size_t sz) const {
    return used_ + GetAlignedMallocSize(sz) <= size_;
  }
  void *GetBase() const { return base_; }
  size_t GetSize() const { return size_; }

 protected:
  // aligned to 8 byte
  // otherwise, a misaligned memory address can cause execution error
  static size_t GetAlignedMallocSize(size_t size) {
    return ((size + 7) >> 3) << 3;
  }

  void *base_;
  size_t size_;
  size_t used_;
};

class NoFreeCacheAllocator : public CacheAllocator {
 public:
  NoFreeCacheAllocator() : CacheAllocator() {}
  NoFreeCacheAllocator(void *b, size_t size) : CacheAllocator(b, size) {}

  void *Malloc(size_t size) {
    size_t aligned_size = GetAlignedMallocSize(size);
    assert(IsAvailable(aligned_size));
    void *ret = static_cast<char *>(base_) + used_;
    used_ += aligned_size;
    return ret;
  }
  // we don't have a Free function
};

class LinearCacheAllocator : public CacheAllocator {
 public:
  LinearCacheAllocator() : CacheAllocator() {}
  LinearCacheAllocator(void *b, size_t size) : CacheAllocator(b, size) {}

  void *Malloc(size_t size) {
    size_t aligned_size = GetAlignedMallocSize(size);
    assert(IsAvailable(aligned_size));
    void *ret = static_cast<char *>(base_) + used_;
    used_ += aligned_size;
    alloc_ptrs_.push(ret);
    alloc_sizes_.push(aligned_size);
    return ret;
  }

  // The memory is allocated and deallocated in LIFO fashion.
  // Will panic if that does not happen.
  // Return false if the memory is not allocated by this allocator.
  bool Free(void *p) {
    if (base_ <= p && p < static_cast<char *>(base_) + size_) {
      assert(alloc_ptrs_.top() == p);
      alloc_ptrs_.pop();
      size_t allocated_size = alloc_sizes_.top();
      alloc_sizes_.pop();
      assert(used_ >= allocated_size);
      used_ -= allocated_size;
      return true;
    }
    return false;
  }

 private:
  std::stack<void *> alloc_ptrs_;
  std::stack<size_t> alloc_sizes_;
};

#endif
