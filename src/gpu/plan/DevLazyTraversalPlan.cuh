#ifndef __GPU_DEV_LAZY_TRAVERSAL_PLAN_CUH__
#define __GPU_DEV_LAZY_TRAVERSAL_PLAN_CUH__

#include "DevTraversalPlan.cuh"
#include "LazyTraversalPlan.h"

class DevLazyTraversalPlan : public DevTraversalPlan {
 public:
  DevLazyTraversalPlan(LazyTraversalPlan* plan, CudaContext* context)
      : DevTraversalPlan(plan, context) {
    AllConnType backward_conn;
    Plan::GetBackwardConnectivity(backward_conn, plan->GetConnectivity(),
                                  plan->GetSearchSequence());
    AllCondType computed_cond;
    plan->GetComputeCondition(computed_cond);
    AllCondType materialized_cond;
    plan->GetMaterializeCondition(materialized_cond);
    AllCondType filter_cond;
    plan->GetFilterCondition(filter_cond);
    AllCondType count_to_materialized_cond;
    LazyTraversalPlanUtils::GetCountToMaterializedVerticesCondition(
        count_to_materialized_cond, plan->GetExecuteOperations(),
        plan->GetOrdering(), plan->GetVertexCount());

    MultiVTGroup& all_materialized_vertices = plan->GetMaterializedVertices();
    MultiVTGroup& all_computed_unmaterialized_vertices =
        plan->GetComputedUnmaterializedVertices();

    Init(backward_conn, computed_cond, materialized_cond, filter_cond,
         count_to_materialized_cond, all_materialized_vertices,
         all_computed_unmaterialized_vertices, plan->GetVertexCount(), context);
  }

  DevLazyTraversalPlan(AllConnType& backward_conn, AllCondType& computed_cond,
                       AllCondType& materialized_cond, AllCondType& filter_cond,
                       AllCondType& count_to_materialized_cond,
                       MultiVTGroup& all_materialized_vertices,
                       MultiVTGroup& all_computed_unmaterialized_vertices,
                       // for DevTraversalPlan
                       AllConnType& ordered_conn,
                       AllCondType& whole_ordered_cond, size_t levels_num,
                       CudaContext* context)
      : DevTraversalPlan(ordered_conn, whole_ordered_cond, levels_num,
                         context) {
    Init(backward_conn, computed_cond, materialized_cond, filter_cond,
         count_to_materialized_cond, all_materialized_vertices,
         all_computed_unmaterialized_vertices, levels_num, context);
  }

  ~DevLazyTraversalPlan() {
    delete backward_conn_;
    backward_conn_ = NULL;
    delete computed_order_;
    computed_order_ = NULL;
    delete materialized_order_;
    materialized_order_ = NULL;
    delete filter_order_;
    filter_order_ = NULL;
    delete count_to_materialized_order_;
    count_to_materialized_order_ = NULL;
    delete materialized_vertices_;
    materialized_vertices_ = NULL;
    delete computed_unmaterialized_vertices_;
    computed_unmaterialized_vertices_ = NULL;
  }

  // getter
  DeviceArray<DevConnType>* GetBackwardConnectivity() const {
    return backward_conn_;
  }
  DeviceArray<DevCondArrayType>* GetComputedOrdering() const {
    return computed_order_;
  }
  DeviceArray<DevCondArrayType>* GetMaterializedOrdering() const {
    return materialized_order_;
  }
  DeviceArray<DevCondArrayType>* GetFilterOrdering() const {
    return filter_order_;
  }
  DeviceArray<DevCondArrayType>* GetCountToMaterializedOrdering() const {
    return count_to_materialized_order_;
  }
  DeviceArray<DevConnType>* GetMaterializedVertices() const {
    return materialized_vertices_;
  }
  DeviceArray<DevConnType>* GetComputedUnmaterializedVertices() const {
    return computed_unmaterialized_vertices_;
  }

  virtual void Print() const {
    DevTraversalPlan::Print();

    size_t n = backward_conn_->GetSize();
    DevConnType* h_backward_conn = new DevConnType[n];
    DevCondArrayType* h_computed_order = new DevCondArrayType[n];
    DevCondArrayType* h_materialized_order = new DevCondArrayType[n];
    DevCondArrayType* h_filter_order = new DevCondArrayType[n];
    DevCondArrayType* h_count_to_materialized_order = new DevCondArrayType[n];
    DToH(h_backward_conn, backward_conn_->GetArray(), n);
    DToH(h_computed_order, computed_order_->GetArray(), n);
    DToH(h_materialized_order, materialized_order_->GetArray(), n);
    DToH(h_filter_order, filter_order_->GetArray(), n);
    DToH(h_count_to_materialized_order,
         count_to_materialized_order_->GetArray(), n);

    for (size_t i = 0; i < n; ++i) {
      std::cout << "DevLazyTraversalPlan, level " << i << ":";
      std::cout << "backward_conn:";
      h_backward_conn[i].Print();
      std::cout << ",computed_order:";
      h_computed_order[i].Print();
      std::cout << ",materialized_order:";
      h_materialized_order[i].Print();
      std::cout << ",filter_order:";
      h_filter_order[i].Print();
      std::cout << ",count_to_materialized_order:";
      h_count_to_materialized_order[i].Print();
      std::cout << std::endl;
    }

    size_t m = materialized_vertices_->GetSize();
    DevConnType* h_materialized_vertices = new DevConnType[m];
    DevConnType* h_computed_unmaterialized_vertices = new DevConnType[m];
    DToH(h_materialized_vertices, materialized_vertices_->GetArray(), m);
    DToH(h_computed_unmaterialized_vertices,
         computed_unmaterialized_vertices_->GetArray(), m);

    for (size_t i = 0; i < m; ++i) {
      std::cout << "DevLazyTraversalPlan: exec_level " << i << ":";
      std::cout << "materialized_vertices:";
      h_materialized_vertices[i].Print();
      std::cout << ",computed_unmaterialized_vertices:";
      h_computed_unmaterialized_vertices[i].Print();
      std::cout << std::endl;
    }

    delete[] h_materialized_vertices;
    h_materialized_vertices = NULL;
    delete[] h_computed_unmaterialized_vertices;
    h_computed_unmaterialized_vertices = NULL;

    delete[] h_backward_conn;
    h_backward_conn = NULL;
    delete[] h_computed_order;
    h_computed_order = NULL;
    delete[] h_materialized_order;
    h_materialized_order = NULL;
    delete[] h_filter_order;
    h_filter_order = NULL;
    delete[] h_count_to_materialized_order;
    h_count_to_materialized_order = NULL;
  }

 private:
  void Init(AllConnType& all_conn, AllCondType& all_computed_order,
            AllCondType& all_materialized_order, AllCondType& all_filter_order,
            AllCondType& all_count_to_materialized_order,
            MultiVTGroup& all_materialized_vertices,
            MultiVTGroup& all_computed_unmaterialized_vertices,
            size_t levels_num, CudaContext* context) {
    backward_conn_ = new DeviceArray<DevConnType>(levels_num, context);
    computed_order_ = new DeviceArray<DevCondArrayType>(levels_num, context);
    materialized_order_ =
        new DeviceArray<DevCondArrayType>(levels_num, context);
    filter_order_ = new DeviceArray<DevCondArrayType>(levels_num, context);
    count_to_materialized_order_ =
        new DeviceArray<DevCondArrayType>(levels_num, context);

    for (size_t i = 0; i < levels_num; ++i) {
      DevConnType h_backward_conn;
      h_backward_conn.Set(all_conn[i]);
      DevCondArrayType h_computed_order;
      h_computed_order.Set(all_computed_order[i]);
      DevCondArrayType h_materialized_order;
      h_materialized_order.Set(all_materialized_order[i]);
      DevCondArrayType h_filter_order;
      h_filter_order.Set(all_filter_order[i]);
      DevCondArrayType h_count_to_materialized_order;
      h_count_to_materialized_order.Set(all_count_to_materialized_order[i]);

      HToD(backward_conn_->GetArray() + i, &h_backward_conn, 1);
      HToD(computed_order_->GetArray() + i, &h_computed_order, 1);
      HToD(materialized_order_->GetArray() + i, &h_materialized_order, 1);
      HToD(filter_order_->GetArray() + i, &h_filter_order, 1);
      HToD(count_to_materialized_order_->GetArray() + i,
           &h_count_to_materialized_order, 1);
    }

    size_t exec_seq_num = all_materialized_vertices.size();
    materialized_vertices_ =
        new DeviceArray<DevConnType>(exec_seq_num, context);
    computed_unmaterialized_vertices_ =
        new DeviceArray<DevConnType>(exec_seq_num, context);
    for (size_t i = 0; i < exec_seq_num; ++i) {
      DevConnType h_materialized_vertices;
      h_materialized_vertices.Set(all_materialized_vertices[i]);
      DevConnType h_computed_unmaterialized_vertices;
      h_computed_unmaterialized_vertices.Set(
          all_computed_unmaterialized_vertices[i]);

      HToD(materialized_vertices_->GetArray() + i, &h_materialized_vertices, 1);
      HToD(computed_unmaterialized_vertices_->GetArray() + i,
           &h_computed_unmaterialized_vertices, 1);
    }
  }

 protected:
  DeviceArray<DevConnType>* backward_conn_;

  // The ordering with those vertices that have been materialized when this
  // vertex is computed. Used by the way of checking connectivity constraints.
  DeviceArray<DevCondArrayType>* computed_order_;

  // The ordering with those vertices that have been materialized
  // after this vertex is computed and before this vertex is materialized.
  // Used when this vertex is materialized.
  DeviceArray<DevCondArrayType>* materialized_order_;

  // the ordering with those vertices that have been materialized
  // after this vertex is computed and before this vertex is filter-computed.
  // Used when this vertex is filter-computed.
  DeviceArray<DevCondArrayType>* filter_order_;

  // the ordering constraints to those materialized vertices before
  // counting in the end.
  // Needed for counting.
  DeviceArray<DevCondArrayType>* count_to_materialized_order_;

  DeviceArray<DevConnType>* materialized_vertices_;
  DeviceArray<DevConnType>* computed_unmaterialized_vertices_;
};

#endif
