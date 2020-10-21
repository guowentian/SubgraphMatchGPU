#ifndef __CPU_BFS_UTILS_H__
#define __CPU_BFS_UTILS_H__

#include <algorithm>
#include <vector>
#include "LinearMemoryAllocator.h"

struct BatchSpec {
  BatchSpec() {}
  BatchSpec(size_t start, size_t end) : start_(start), end_(end) {}
  size_t GetSize() const { return end_ - start_; }
  size_t GetStart() const { return start_; }
  size_t GetEnd() const { return end_; }

  // [start_, end_)
  size_t start_;
  size_t end_;
};

template <typename T>
class Array {
 public:
  Array() {
    array_ = NULL;
    size_ = 0;
  }
  Array(T* arr, size_t sz) : array_(arr), size_(sz) {}
  //~Array() { assert(array_ == 0); }

  void Init(LinearMemoryAllocator* alloc, size_t sz) {
    array_ = static_cast<T*>(alloc->SafeAllocate(sz * sizeof(T)));
    size_ = sz;
  }
  void Init(T* arr, size_t sz) {
    array_ = arr;
    size_ = sz;
  }
  void Release(LinearMemoryAllocator* alloc) {
    alloc->Release(array_);
    clear();
  }
  size_t GetMemoryCost() const { return sizeof(T) * size_; }

  void clear() {
    array_ = NULL;
    size_ = 0;
  }
  size_t size() const { return size_; }
  T* data() { return array_; }
  T& operator[](size_t idx) { return array_[idx]; }
  const T operator[](size_t idx) const { return array_[idx]; }

 private:
  T* array_;
  size_t size_;
};

template <typename T>
using LayeredArray = std::vector<Array<T> >;

// A number of arrays,
// each has predefined size, and a counter to track the current length
template <typename T>
class SegmentedArray {
 public:
  SegmentedArray() {
    offsets_ = NULL;
    segments_num_ = 0;
  }

  void Init(LinearMemoryAllocator* alloc, size_t* offsets,
            size_t segments_num) {
    offsets_ = offsets;
    segments_num_ = segments_num;
    size_t total_size = offsets_[segments_num_];
    arrays_.Init(alloc, total_size);
    sizes_.Init(alloc, total_size);
  }

  // The user is responsible to clear up memory after usage
  void Release(LinearMemoryAllocator* alloc) {
    offsets_ = 0;
    segments_num_ = 0;
    sizes_.Release(alloc);
    arrays_.Release(alloc);
  }

  // Get data
  Array<T>& GetArrays() { return arrays_; }
  Array<size_t>& GetSizes() { return sizes_; }

  // manipulate the arrays of many segments
  void SetSize(size_t seg_id, size_t sz) { sizes_[seg_id] = sz; }
  T* GetArray(size_t segment_id) {
    return arrays_.data() + offsets_[segment_id];
  }
  size_t GetSegmentsNum() const { return segments_num_; }
  size_t GetSize(size_t segment_id) const { return sizes_[segment_id]; }
  size_t GetSizeBound(size_t segment_id) const {
    return offsets_[segment_id + 1] - offsets_[segment_id];
  }

  size_t GetMemoryCost() const {
    return arrays_.GetMemoryCost() + sizes_.GetMemoryCost();
  }

 private:
  size_t* offsets_;
  size_t segments_num_;

  Array<T> arrays_;
  // be careful that sizes_ are unintialized.
  // The user is responsible to assign size by themselves
  Array<size_t> sizes_;
};

template <typename T>
using LayeredSegmentedArray = std::vector<SegmentedArray<T> >;

static void OrganizeBatch(size_t* children_offset, size_t parent_count,
                          size_t children_count_per_batch,
                          std::vector<BatchSpec>& batches) {
  size_t total_children_count = children_offset[parent_count];
  size_t batch_num = (total_children_count + children_count_per_batch - 1) /
                     children_count_per_batch;
  size_t cur_parent_id = 0;
  for (size_t batch_id = 0; batch_id < batch_num; ++batch_id) {
    size_t batch_end = (batch_id + 1) * children_count_per_batch;
    size_t p = std::lower_bound(children_offset + cur_parent_id,
                                children_offset + parent_count, batch_end) -
               (children_offset + cur_parent_id);
    // ensure increase cur_parent_id
    assert(p);
    batches.push_back(BatchSpec(cur_parent_id, cur_parent_id + p));
    cur_parent_id += p;
  }
  assert(batches[batch_num - 1].GetEnd() == parent_count);
}

#endif
