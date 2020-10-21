#ifndef __GPU_FILTER_CUH__
#define __GPU_FILTER_CUH__

#include <cassert>
#include <cub/cub.cuh>
#include "CudaContext.cuh"
#include "Meta.h"
#include "Transform.cuh"

namespace GpuUtils {
namespace Filter {

template <typename uintV>
DEVICE bool CheckVertexCondition(uintV first_vertex_id, uintV second_vertex_id,
                                 CondOperator op) {
  switch (op) {
    case LESS_THAN:
      return first_vertex_id < second_vertex_id;
    case LARGER_THAN:
      return first_vertex_id > second_vertex_id;
    case NON_EQUAL:
      return first_vertex_id != second_vertex_id;
    default:
      return true;
  }
}

}  // namespace Filter
}  // namespace GpuUtils
#endif
