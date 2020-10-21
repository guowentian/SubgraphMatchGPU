#ifndef __GPU_DEV_TRAVERSAL_PLAN_CUH__
#define __GPU_DEV_TRAVERSAL_PLAN_CUH__

#include "DevPlan.cuh"
#include "TraversalPlan.h"

class DevTraversalPlan {
 public:
  DevTraversalPlan(TraversalPlan* plan, CudaContext* context)
      : backward_conn_(NULL), backward_cond_(NULL) {
    AllConnType ordered_conn;
    plan->GetOrderedConnectivity(ordered_conn);
    AllCondType whole_ordered_cond;
    plan->GetWholeOrderedOrdering(whole_ordered_cond);
    Init(ordered_conn, whole_ordered_cond, plan->GetVertexCount(), context);
  }
  DevTraversalPlan(AllConnType& conn, AllCondType& cond, size_t levels_num,
                   CudaContext* context)
      : backward_conn_(NULL), backward_cond_(NULL) {
    Init(conn, cond, levels_num, context);
  }
  ~DevTraversalPlan() {
    ReleaseIfExists(backward_conn_);
    ReleaseIfExists(backward_cond_);
  }

  virtual void Print() const {
    std::cout << "DevTraversalPlan, connectivity:";
    PrintDeviceArray<DevConnType>(backward_conn_);

    std::cout << "DevTraversalPlan, condition:";
    PrintDeviceArray<DevCondArrayType>(backward_cond_);
  }

  // ========== getter =============
  size_t GetLevelsNum() const { return levels_num_; }
  DeviceArray<DevConnType>* GetBackwardConnectivity() const {
    return backward_conn_;
  }
  DeviceArray<DevCondArrayType>* GetBackwardCondition() const {
    return backward_cond_;
  }

 private:
  void Init(AllConnType& conn, AllCondType& cond, size_t levels_num,
            CudaContext* context) {
    levels_num_ = levels_num;
    ReAllocate(backward_conn_, levels_num_, context);
    ReAllocate(backward_cond_, levels_num_, context);

    for (size_t i = 0; i < levels_num_; ++i) {
      DevConnType h_conn;
      h_conn.Set(conn[i]);
      HToD(backward_conn_->GetArray() + i, &h_conn, 1);
      DevCondArrayType h_cond;
      h_cond.Set(cond[i]);
      HToD(backward_cond_->GetArray() + i, &h_cond, 1);
    }
  }

 protected:
  size_t levels_num_;
  DeviceArray<DevConnType>* backward_conn_;
  DeviceArray<DevCondArrayType>* backward_cond_;
};

#endif
