#ifndef __SPLIT_BATCH_CUH__
#define __SPLIT_BATCH_CUH__

#include "GPUBinSearch.cuh"
#include "Meta.h"
#include "Transform.cuh"
/**
 * \brief Determine the right end point for each batch
 * \details Input: a sequence of parents, each has a number of children,
 * and each child is associated with a cost.
 * Based on the prefix sum of the costs, determine
 * the batches: [L_0, R_0), [R_0, R_1), [R1, R_2) ... [R_n-2, R_n-1)
 * where n=batch_num.
 * Since L_0 is (0,0), here we are looking for each R_i
 * Each end point of the batch is represented as
 * a pair (parent_index, children_index).
 * Each achieved batch has the memory size of batch_size, with the error
 * [0, memory cost of two children)
 * Output:
 * \tparam  IndexType
 * \tparam  size_t
 * \param[in]  batch_parent_indices_end :
 * record the parent index of the end point of the batch.
 * The value is always in the range of [0,parent_count-1]
 * \param[in]  batch_parent_cost_prefix_sum :
 * record the cost for each batch
 * \param[in]  batch_children_indices_end :
 * record the children_index of the end point of the batch
 * \param[in]  batch_size
 * \param[in]  batch_num
 * \param[in]  parent_cost_prefix_sum
 * \param[in]  children_count
 * \param[in]  parent_count
 * \exception
 */

template <typename IndexType, typename size_t>
__global__ void SplitBatchKernel(IndexType *batch_parent_indices_end,
                                 size_t *batch_parent_cost_prefix_sum,
                                 size_t *batch_children_indices_end,
                                 size_t batch_size, size_t batch_num,
                                 size_t *parent_cost_prefix_sum,
                                 size_t *children_count,
                                 IndexType parent_count) {
  assert(batch_num * batch_size >= parent_cost_prefix_sum[parent_count]);
  size_t cta_offset = blockIdx.x * blockDim.x;
  while (cta_offset < batch_num) {
    size_t batch_id = cta_offset + threadIdx.x;
    if (batch_id < batch_num) {
      if (batch_id + 1 == batch_num) {
        batch_parent_indices_end[batch_id] = parent_count - 1;
        batch_parent_cost_prefix_sum[batch_id] =
            parent_cost_prefix_sum[parent_count];
        batch_children_indices_end[batch_id] = children_count[parent_count - 1];
      } else {
        size_t target = batch_size * (batch_id + 1);
        IndexType pos = GpuUtils::BinSearch::BinSearch<IndexType, size_t>(
            parent_cost_prefix_sum, parent_count + 1, target);
        assert(pos <= parent_count);
        if (parent_cost_prefix_sum[pos] > target) {
          assert(pos >= 1);
          batch_parent_indices_end[batch_id] = pos - 1;
          double cost_per_child =
              (parent_cost_prefix_sum[pos] - parent_cost_prefix_sum[pos - 1]) *
              1.0 / children_count[pos - 1];
          batch_children_indices_end[batch_id] =
              (batch_size * (batch_id + 1) - parent_cost_prefix_sum[pos - 1]) /
              cost_per_child;
          batch_parent_cost_prefix_sum[batch_id] =
              parent_cost_prefix_sum[pos - 1] +
              batch_children_indices_end[batch_id] * cost_per_child;
        } else if (parent_cost_prefix_sum[pos] == target) {
          assert(pos >= 1);
          batch_parent_indices_end[batch_id] = pos - 1;
          batch_parent_cost_prefix_sum[batch_id] = parent_cost_prefix_sum[pos];
          ;
          batch_children_indices_end[batch_id] = children_count[pos - 1];
        } else {
          // The case for parent_cost_prefix_sum[pos] < target
          // only happens in the last batch
          assert(false);
        }
      }
    }
    cta_offset += blockDim.x * gridDim.x;
  }
}

template <typename IndexType, typename CostType>
static void SplitBatch(IndexType *batch_indices_end, CostType batch_size,
                       size_t batch_num, size_t parent_count,
                       CostType *batch_cost_prefix_sum, CudaContext *context) {
  GpuUtils::Transform::Transform(
      [=] DEVICE(int batch_id) {
        if (batch_id + 1 == batch_num) {
          batch_indices_end[batch_id] = parent_count - 1;
        } else {
          CostType target = (batch_id + 1) * batch_size;
          IndexType pos = GpuUtils::BinSearch::BinSearch(
              batch_cost_prefix_sum, parent_count + 1, target);
          batch_indices_end[batch_id] = pos;
        }

      },
      batch_num, context);
}

#endif
