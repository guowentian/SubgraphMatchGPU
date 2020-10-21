#ifndef __EXTERNAL_LIGHT_COMMON_CUH__
#define __EXTERNAL_LIGHT_COMMON_CUH__

#include "LIGHTCommon.cuh"

namespace Light {
// Given the path id, find the set of backward neighbors vertices.
// Write the results in vertices and return the count.
// Used for batching and load subgraphs.
struct BackwardNeighborGatherFunctor {
  DevConnType *conn_;
  uintV **d_seq_instances_;

  BackwardNeighborGatherFunctor(DevConnType *conn, uintV **d_seq_instances) {
    conn_ = conn;
    d_seq_instances_ = d_seq_instances;
  }
  HOST_DEVICE
  BackwardNeighborGatherFunctor(const BackwardNeighborGatherFunctor &rht) {
    conn_ = rht.conn_;
    d_seq_instances_ = rht.d_seq_instances_;
  }

  DEVICE size_t operator()(size_t path_id,
                           uintV vertices[kMaxQueryVerticesNum]) const {
    for (size_t i = 0; i < conn_->GetCount(); ++i) {
      uintV u = conn_->Get(i);
      uintV v = d_seq_instances_[u][path_id];
      vertices[i] = v;
    }
    return conn_->GetCount();
  }
};
}  // namespace Light

#endif