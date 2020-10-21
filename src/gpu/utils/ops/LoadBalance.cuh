#ifndef __GPU_UTILS_LOAD_BALANCE_CUH__
#define __GPU_UTILS_LOAD_BALANCE_CUH__

#include <moderngpu/kernel_load_balance.hxx>
#include "CudaContext.cuh"
#include "MGPUContext.cuh"

namespace GpuUtils {
namespace LoadBalance {

// ================= entry =======================

// paremeter of func (int index, int seg, int intra_seg_rank,
// ...)
template <typename launch_arg_t = mgpu::empty_t, typename Func,
          typename SegmentIterator>
void LBSTransform(Func func, int count, SegmentIterator segments,
                  int num_segments, CudaContext* context) {
  typedef
      typename mgpu::conditional_typedef_t<launch_arg_t, MGPULaunchBox>::type_t
          LaunchBox;
  mgpu::transform_lbs<LaunchBox>(func, count, segments, num_segments, *context);
}

template <typename launch_arg_t = mgpu::empty_t, typename SegmentIterator,
          typename OutputIterator>
void LoadBalanceSearch(int count, SegmentIterator segments, int num_segments,
                       OutputIterator output, CudaContext* context) {
  typedef
      typename mgpu::conditional_typedef_t<launch_arg_t, MGPULaunchBox>::type_t
          LaunchBox;
  mgpu::load_balance_search<LaunchBox>(count, segments, num_segments, output,
                                       *context);
}
}  // namespace LoadBalance
}  // namespace GpuUtils

#endif
