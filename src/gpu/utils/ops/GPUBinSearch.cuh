#ifndef __GPU_BINARY_SEARCH_CUH__
#define __GPU_BINARY_SEARCH_CUH__

namespace GpuUtils {
namespace BinSearch {
enum BoundType { BOUND_LOWER, BOUND_UPPER };
/**
 * \brief Get the smallest position of the element that is no less than k
 * \details
 * \tparam  IndexType
 * \tparam  DataType
 * \param[in]  arr
 * \param[in]  len 0-based
 * \param[in]  k
 * \return If no such value exists, return len-1
 * \exception
 */
template <typename IndexType, typename DataType, typename DataIt,
          BoundType bound_type = BOUND_LOWER>
__device__ IndexType BinSearch(DataIt arr, IndexType len, DataType k) {
  IndexType left = 0, right = len;
  while (left < right) {
    IndexType mid = (left + right) / 2;
    bool pred = (bound_type == BOUND_LOWER) ? arr[mid] < k : !(k < arr[mid]);
    if (pred)
      left = mid + 1;
    else
      right = mid;
  }
  return left;
}
}  // namespace BinSearch
}  // namespace GpuUtils

#endif
