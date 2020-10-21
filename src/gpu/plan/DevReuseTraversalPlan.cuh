#ifndef __GPU_DEV_REUSE_TRAVERSAL_PLAN_CUH__
#define __GPU_DEV_REUSE_TRAVERSAL_PLAN_CUH__

#include "DevTraversalPlan.cuh"
#include "ReuseTraversalPlan.h"

struct DevReuseConnMeta {
  HOST void Set(ReuseConnMeta& from) {
    conn_.Set(from.conn_);
    source_conn_.Set(from.source_conn_);
    source_vertex_ = from.source_vertex_;
    aligned_vertex_ = from.GetAlignedVertex();
  }

  HOST void Print() const {
    std::cout << "source_vertex=" << source_vertex_
              << ",aligned_level=" << aligned_vertex_ << ",conn=";
    conn_.Print();
    std::cout << "source_conn=";
    source_conn_.Print();
  }

  DEVICE DevConnType& GetConnectivity() { return conn_; }
  DEVICE DevConnType& GetSourceConnectivity() { return source_conn_; }
  DEVICE uintV GetSourceVertex() const { return source_vertex_; }
  DEVICE size_t GetAlignedVertex() const { return aligned_vertex_; }

  DevConnType conn_;
  DevConnType source_conn_;
  uintV source_vertex_;
  size_t aligned_vertex_;
};

struct DevVertexReuseIntersectPlan {
  HOST void Set(VertexReuseIntersectPlan& from) {
    separate_conn_.Set(from.separate_conn_);
    reuse_conn_meta_count_ = from.reuse_conn_meta_.size();
    for (size_t i = 0; i < from.reuse_conn_meta_.size(); ++i) {
      reuse_conn_meta_[i].Set(from.reuse_conn_meta_[i]);
    }
  }

  HOST void Print() const {
    std::cout << "reuse_conn_meta_count=" << reuse_conn_meta_count_;
    for (size_t j = 0; j < reuse_conn_meta_count_; ++j) {
      std::cout << " reuse_conn_meta[" << j << "]=[";
      reuse_conn_meta_[j].Print();
      std::cout << "]";
    }
    std::cout << " ";
    separate_conn_.Print();
    std::cout << std::endl;
  }

  DEVICE DevReuseConnMeta& GetReuseConnectivityMeta(size_t i) {
    return reuse_conn_meta_[i];
  }
  DEVICE size_t GetReuseConnectivityMetaCount() const {
    return reuse_conn_meta_count_;
  }
  DEVICE DevConnType& GetSeparateConnectivity() { return separate_conn_; }

  DevReuseConnMeta reuse_conn_meta_[kMaxQueryVerticesNum];
  size_t reuse_conn_meta_count_;
  DevConnType separate_conn_;
};

class DevReuseTraversalPlan : public DevTraversalPlan {
 public:
  DevReuseTraversalPlan(ReuseTraversalPlan* plan, CudaContext* context)
      : DevTraversalPlan(plan, context) {
    level_reuse_intersect_plan_ =
        new DeviceArray<DevVertexReuseIntersectPlan>(levels_num_, context);

    for (size_t i = 0; i < levels_num_; ++i) {
      VertexReuseIntersectPlan& orig_vertex_plan =
          plan->GetLevelReuseIntersectPlan()[i];
      DevVertexReuseIntersectPlan h_vertex_plan;
      memset(&h_vertex_plan, 0, sizeof(h_vertex_plan));
      h_vertex_plan.Set(orig_vertex_plan);
      HToD(level_reuse_intersect_plan_->GetArray() + i, &h_vertex_plan, 1);

#if defined(DEBUG)
      std::cout << "DevReuseTraversalPlan: level=" << i << ":";
      h_vertex_plan.Print();
      std::cout << std::endl;
#endif
    }
  }
  ~DevReuseTraversalPlan() {
    delete level_reuse_intersect_plan_;
    level_reuse_intersect_plan_ = NULL;
  }

  // getter
  DeviceArray<DevVertexReuseIntersectPlan>* GetLevelReuseIntersectPlan() const {
    return level_reuse_intersect_plan_;
  }

 protected:
  DeviceArray<DevVertexReuseIntersectPlan>* level_reuse_intersect_plan_;
};

#endif
