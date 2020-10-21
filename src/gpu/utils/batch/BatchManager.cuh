#ifndef __GPU_BATCH_MANAGER_CUH__
#define __GPU_BATCH_MANAGER_CUH__

#include <vector>
#include "CudaContext.cuh"
#include "DeviceArray.cuh"
#include "GPUUtil.cuh"
#include "Meta.h"
#include "Scan.cuh"
#include "SplitBatch.cuh"
#include "Transform.cuh"

class BatchSpec {
 public:
  BatchSpec(size_t left, size_t right) {
    batch_left_ = left;
    batch_right_ = right;
  }
  BatchSpec(const BatchSpec &obj) {
    batch_left_ = obj.GetBatchLeftEnd();
    batch_right_ = obj.GetBatchRightEnd();
  }

  size_t GetBatchCount() const { return batch_right_ - batch_left_ + 1; }
  size_t GetBatchLeftEnd() const { return batch_left_; }
  size_t GetBatchRightEnd() const { return batch_right_; }

 private:
  size_t batch_left_;
  size_t batch_right_;
};

class BatchManager {
 public:
  BatchManager(CudaContext *context, size_t batch_size)
      : context_(context), batch_size_(batch_size), batch_num_(0) {}

  /**
   * \param[in]  parent_count : number of elements in total
   * \param[in]  parent_factor : the cost for each element
   */
  void OrganizeBatch(size_t parent_count, size_t parent_factor) {
    size_t total_cost = parent_factor * parent_count;
    batch_num_ = (total_cost + batch_size_ - 1) / batch_size_;
    batch_indices_.resize(batch_num_ + 1);
    size_t batch_element_count = (parent_count + batch_num_ - 1) / batch_num_;
    batch_indices_[0] = 0;
    for (size_t batch_id = 0; batch_id < batch_num_; ++batch_id) {
      batch_indices_[batch_id + 1] = batch_element_count * (batch_id + 1);
    }
    if (batch_indices_[batch_num_] > parent_count - 1) {
      batch_indices_[batch_num_] = parent_count - 1;
    }
  }

  // children_count: for each parent, the number of children it has
  // children_cost: the cost for each parent (in bytes)
  void OrganizeBatch(DeviceArray<size_t> *children_count,
                     DeviceArray<size_t> *children_cost, size_t parent_count,
                     CudaContext *context) {
    DeviceArray<size_t> parent_cost_prefix_sum(parent_count + 1, context);
    GpuUtils::Scan::ExclusiveSum(
        children_cost->GetArray(), parent_count,
        parent_cost_prefix_sum.GetArray(),
        parent_cost_prefix_sum.GetArray() + parent_count, context);
    size_t total_cost = 0;
    DToH(&total_cost, parent_cost_prefix_sum.GetArray() + parent_count, 1);

    // decide batch_num
    batch_num_ = (total_cost + batch_size_ - 1) / batch_size_;
    batch_indices_.resize(batch_num_ + 1);

    // SplitBatchKernel
    DeviceArray<size_t> batch_parent_indices_end(batch_num_, context_);
    DeviceArray<size_t> batch_children_indices_end(batch_num_, context_);
    DeviceArray<size_t> batch_parent_cost_prefix_sum_end(batch_num_, context_);
    SplitBatchKernel<size_t, size_t>
        <<<MAX_BLOCKS_NUM, THREADS_PER_BLOCK, 0, context->Stream()>>>(
            batch_parent_indices_end.GetArray(),
            batch_parent_cost_prefix_sum_end.GetArray(),
            batch_children_indices_end.GetArray(), batch_size_, batch_num_,
            parent_cost_prefix_sum.GetArray(), children_count->GetArray(),
            parent_count);

    // SplitBatch(batch_parent_indices_end.GetArray(), batch_size_, batch_num_,
    //           parent_count, parent_cost_prefix_sum.GetArray(), context);
    DToH(batch_indices_.data() + 1, batch_parent_indices_end.GetArray(),
         batch_num_);
    batch_indices_[0] = 0;

    this->CompactBatch();
  }

  // The cost for each parent:
  // #children_number * children_factor + parent_factor
  void OrganizeBatch(DeviceArray<size_t> *children_count, size_t parent_factor,
                     size_t children_factor, size_t parent_count,
                     CudaContext *context) {
    DeviceArray<size_t> children_cost(parent_count, context);
    size_t *children_count_data = children_count->GetArray();
    size_t *children_cost_data = children_cost.GetArray();
    GpuUtils::Transform::Transform(
        [=] DEVICE(int index) {
          children_cost_data[index] =
              children_count_data[index] * children_factor + parent_factor;
        },
        parent_count, context);

    OrganizeBatch(children_count, &children_cost, parent_count, context);
  }

  BatchSpec GetBatch(size_t batch_id) const {
    // BatchSpec has inclusive interval end
    return BatchSpec(batch_indices_[batch_id],
                     batch_id == batch_num_ - 1
                         ? batch_indices_[batch_id + 1]
                         : batch_indices_[batch_id + 1] - 1);
  }

  size_t GetBatchNum() const { return batch_num_; }
  size_t GetBatchSize() const { return batch_size_; }

  void SetBatches(const std::vector<BatchSpec> &batches) {
    batch_num_ = batches.size();
    batch_indices_.resize(batch_num_ + 1);
    batch_indices_[0] = 0;
    for (size_t batch_id = 0; batch_id < batch_num_; ++batch_id) {
      batch_indices_[batch_id + 1] = batches[batch_id].GetBatchRightEnd() + 1;
    }
    batch_indices_[batch_num_] = batches[batch_num_ - 1].GetBatchRightEnd();
  }

  static size_t GetSafeBatchSize(size_t batch_size) {
    // conservative usage
    batch_size *= 0.9;
    // To ensure LBSTransform can execute normally,
    // note that at cta_merge.hxx : 15, 'begin+end' may get INT overflow.
    // As begin<=diag and end<=diag, we can force diag<=(1<<30)-1
    batch_size = std::min(batch_size, (size_t)((1ULL << 30) - 1) * sizeof(int));
    // at least 128MB
    batch_size = std::max(batch_size, (size_t)(1ULL << 20) * 128);
    return batch_size;
  }

 private:
  // After SplitBatch, batch_indices_ could be like 0,0,1,1,...
  // CompactBatch() ensures each batch has at least one element to process
  void CompactBatch() {
    std::vector<size_t> new_batches;
    size_t final_batch_end = batch_indices_.back();
    new_batches.push_back(0);
    for (size_t i = 1; i < batch_indices_.size(); ++i) {
      assert(batch_indices_[i - 1] <= batch_indices_[i]);
      new_batches.push_back(batch_indices_[i - 1] == batch_indices_[i]
                                ? batch_indices_[i - 1] + 1
                                : batch_indices_[i]);
      if (new_batches.back() >= final_batch_end) {
        // reach the last element already
        if (new_batches.back() > final_batch_end) {
          // only when the total range of parent_indices is [0,0]
          assert(final_batch_end == 0);
          new_batches[i] = final_batch_end;
        }
        break;
      }
    }

    batch_num_ = new_batches.size() - 1;
    batch_indices_.swap(new_batches);
  }

 protected:
  CudaContext *context_;
  size_t batch_size_;
  size_t batch_num_;
  std::vector<size_t> batch_indices_;
};

#endif
