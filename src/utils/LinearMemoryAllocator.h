#ifndef __UTILS_LINEAR_MEMORY_ALLOCATOR_H__
#define __UTILS_LINEAR_MEMORY_ALLOCATOR_H__

#include <cassert>
#include <stack>

// Memory allocated from pre-allocated cache.
// Memory is required to be released in the order of allocation,
// which can ease the memory recycle.
class LinearMemoryAllocator {
 public:
  LinearMemoryAllocator(size_t buffer_limit) : buffer_limit_(buffer_limit) {
    buffer_size_ = 0;
    memory_ = new char[buffer_limit_];
  }
  ~LinearMemoryAllocator() {
    delete[] memory_;
    memory_ = NULL;
  }

  // Allocate memory from buffer. If insufficient memory, return NULL
  virtual void* Allocate(size_t sz) {
    // ensure each allocation aligns to multiple of word size
    sz = ((sz + 7) >> 3) << 3;
    if (AvailableMemory(sz)) {
      char* ret = memory_ + buffer_size_;
      buffer_size_ += sz;

      buf_alloc_ptrs_.push(ret);
      buf_alloc_sizes_.push(sz);

      return ret;
    } else {
      return NULL;
    }
  }

  // Allocate memory from buffer first, if insufficient memory in buffer,
  // then allocate from system memory
  virtual void* SafeAllocate(size_t sz) {
    void* ret = Allocate(sz);
    if (ret == NULL) {
      std::cout << "insufficient buffer !" << std::endl;
      ret = new char[sz];
    }
    return ret;
    //   return new char[sz];
  }

  virtual void Release(void* ptr) {
    char* ch_ptr = static_cast<char*>(ptr);
    if (memory_ <= ch_ptr && ch_ptr < memory_ + buffer_limit_) {
      // If it was allocated from buffer
      assert(ptr == buf_alloc_ptrs_.top());
      buf_alloc_ptrs_.pop();
      size_t release_size = buf_alloc_sizes_.top();
      buf_alloc_sizes_.pop();
      assert(buffer_size_ >= release_size);
      buffer_size_ -= release_size;
    } else {
      std::cout << "Release system memory !" << std::endl;
      delete[] ch_ptr;
      ch_ptr = NULL;
    }
    // delete[] ch_ptr;
  }

  size_t GetBufferSize() const { return buffer_size_; }
  size_t GetBufferLimit() const { return buffer_limit_; }
  size_t GetAvailableSize() const { return buffer_limit_ - buffer_size_; }

 private:
  bool AvailableMemory(size_t sz) { return buffer_size_ + sz <= buffer_limit_; }

 private:
  size_t buffer_limit_;
  size_t buffer_size_;
  char* memory_;

  std::stack<void*> buf_alloc_ptrs_;
  std::stack<size_t> buf_alloc_sizes_;
};

// Inherit the same API, for testing purpose
class DirectMemoryAllocator : public LinearMemoryAllocator {
 public:
  DirectMemoryAllocator(size_t limit) : LinearMemoryAllocator(limit) {}

  void* Allocate(size_t sz) { return new char[sz]; }

  void* SafeAllocate(size_t sz) { return Allocate(sz); }

  void Release(void* ptr) {
    char* ch_ptr = static_cast<char*>(ptr);
    delete[] ch_ptr;
    ch_ptr = NULL;
  }
};
#endif
