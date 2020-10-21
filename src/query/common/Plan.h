#ifndef __PLAN_H__
#define __PLAN_H__

#include "Query.h"

class Plan {
 public:
  Plan(Query* query, size_t h_part_num, size_t dev_num, ExecuteMode mode,
       Variant variant)
      : query_(query),
        h_part_num_(h_part_num),
        dev_num_(dev_num),
        execute_mode_(mode),
        variant_(variant) {
    d_part_num_ = dev_num_;
  }
  ~Plan() {}

  Query* GetQuery() const { return query_; }
  size_t GetHostPartitionNum() const { return h_part_num_; }
  size_t GetDeviceNum() const { return dev_num_; }
  size_t GetDevPartitionNum() const { return d_part_num_; }
  ExecuteMode GetExecuteMode() const { return execute_mode_; }
  Variant GetVariant() const { return variant_; }

  virtual void OptimizePlan() = 0;
  virtual void Print() const = 0;

  // Based on the given 'order', get the condition operator between
  // l1 and l2. If there is no specified operator between them,
  // just set it to NON_EQUAL.
  static CondOperator GetConditionType(uintV l1, uintV l2,
                                       const AllCondType& order) {
    CondOperator op = NON_EQUAL;
    for (size_t cond_id = 0; cond_id < order[l1].size(); ++cond_id) {
      CondType cond = order[l1][cond_id];
      if (cond.second == l2) {
        op = cond.first;
        break;
      }
    }
    return op;
  }

  // based on the search sequence, obtain the ranking for each vertex,
  // i.e., the mapping: old vertex id -> new vertex id.
  // A helper function frequently used.
  static void GetInvertedMap(SearchSequence& ret, const SearchSequence& seq) {
    size_t n = seq.size();
    ret.resize(n);
    for (uintV l = 0; l < n; ++l) {
      auto u = seq[l];
      ret[u] = l;
    }
  }

  // Based on the overall connectivity given by con, extract the backward
  // neighbors for each pattern vertex. The backward neighbors of u are those
  // neighbors that have smaller vertex ids than u.
  static void GetOrderedConnectivity(AllConnType& ret, const AllConnType& con) {
    auto n = con.size();
    ret.resize(n);
    for (uintV u = 0; u < n; ++u) {
      ret[u].clear();
      for (size_t j = 0; j < con[u].size(); ++j) {
        if (con[u][j] < u) {
          ret[u].push_back(con[u][j]);
        }
      }
    }
  }
  // Based on the overall ordering given by order, extract the orders between u
  // and u's backward neighbors.
  // The backward neighbors of u are those neighbors that have smaller vertex
  // ids
  // than u.
  static void GetOrderedOrdering(AllCondType& ret, const AllCondType& order) {
    auto n = order.size();
    ret.resize(n);
    for (uintV u = 0; u < n; ++u) {
      ret[u].clear();
      for (size_t j = 0; j < order[u].size(); ++j) {
        if (order[u][j].second < u) {
          ret[u].push_back(order[u][j]);
        }
      }
    }
  }

  // Given the search sequence, extract the original vertex ids in conn
  // with the corresponding index in the search sequence.
  static void GetIndexBasedConnectivity(AllConnType& ret,
                                        const AllConnType& conn,
                                        const SearchSequence& seq) {
    size_t n = seq.size();
    SearchSequence otn_map;
    GetInvertedMap(otn_map, seq);

    ret.resize(n);
    for (uintV l = 0; l < n; ++l) {
      ret[l].clear();
      auto u = seq[l];
      for (auto v : conn[u]) {
        auto nv = otn_map[v];
        ret[l].push_back(nv);
      }
    }
  }

  // Given the search sequence, replace the original vertex ids in order with
  // the corresponding index in the search sequence.
  static void GetIndexBasedOrdering(AllCondType& ret, const AllCondType& order,
                                    const SearchSequence& seq) {
    size_t n = seq.size();
    SearchSequence otn_map;
    GetInvertedMap(otn_map, seq);

    ret.resize(n);
    for (uintV l = 0; l < n; ++l) {
      ret[l].clear();

      auto u = seq[l];
      for (size_t j = 0; j < order[u].size(); ++j) {
        auto nu = otn_map[order[u][j].second];
        ret[l].push_back(std::make_pair(order[u][j].first, nu));
      }
    }
  }

  // Based on the overall connectivity given by con,
  // extract the backward neighbors for each pattern vertex.
  // The new connectivity achieved in 'ret' replaces the original vertex id
  // with its index in the match order 'seq', which is so-called index based.
  // The backward neighbors of u are those neighbors that are positioned
  // before u in the match order.
  static void GetOrderedIndexBasedConnectivity(AllConnType& ret,
                                               const AllConnType& con,
                                               const SearchSequence& seq) {
    AllConnType index_based_conn;
    GetIndexBasedConnectivity(index_based_conn, con, seq);

    GetOrderedConnectivity(ret, index_based_conn);
  }

  // Based on the overall ordering given by 'order',
  // extract the orders between u and u's backward neighbors.
  // The new ordering achieved in 'ret' replaces the original vertex id
  // with its index in the match order 'seq', which is so-called index based.
  // The backward neighbors of u are those neighbors that are positioned
  // before u in the match order.
  static void GetOrderedIndexBasedOrdering(AllCondType& ret,
                                           const AllCondType& order,
                                           const SearchSequence& seq) {
    AllCondType index_based_order;
    GetIndexBasedOrdering(index_based_order, order, seq);

    GetOrderedOrdering(ret, index_based_order);
  }

  // Similar to GetOrderedOrdering, but for those pairs of pattern vertices that
  // do not have condition (<,>), set their conditions as NON_EQUAL.
  // Thus, the resulting 'ret' has n*(n-1)/2 conditions.
  static void GetWholeOrderedOrdering(AllCondType& ret,
                                      const AllCondType& order) {
    size_t n = order.size();
    ret.resize(n);
    for (uintV u = 0; u < n; ++u) {
      ret[u].clear();
      for (uintV v = 0; v < u; ++v) {
        CondOperator cond_op = Plan::GetConditionType(u, v, order);
        ret[u].push_back(std::make_pair(cond_op, v));
      }
    }
  }

  // Similar to GetOrderedIndexBasedOrdering,
  // but for those pairs of pattern vertices that
  // do not have condition (<,>), set their conditions as NON_EQUAL.
  // Thus, the resulting 'ret' has n*(n-1)/2 conditions.
  static void GetWholeOrderedIndexBasedOrdering(AllCondType& ret,
                                                const AllCondType& order,
                                                const SearchSequence& seq) {
    AllCondType index_based_order;
    GetOrderedIndexBasedOrdering(index_based_order, order, seq);
    auto n = order.size();
    ret.resize(n);
    for (uintV i = 0; i < n; ++i) {
      ret[i].clear();
      for (uintV j = 0; j < i; ++j) {
        CondOperator cond_op = Plan::GetConditionType(i, j, index_based_order);
        ret[i].push_back(std::make_pair(cond_op, j));
      }
    }
  }

  // extract the backward neighbors and stores the results in ret.
  // The backward neighbors of u are those neighbors of u that appear
  // before u in the match order.
  // The difference with GetOrderedConnectivity is that we check the 'backward'
  // by the match order 'seq' and we store the results with the original
  // vertex id (i.e., not index-based and not based on vertex ids).
  static void GetBackwardConnectivity(AllConnType& ret, const AllConnType& con,
                                      const SearchSequence& seq) {
    auto n = con.size();
    ret.resize(n);
    std::vector<bool> vis(n, false);
    for (uintV i = 0; i < n; ++i) {
      auto u = seq[i];
      ret[u].clear();
      for (auto nu : con[u]) {
        if (vis[nu]) {
          ret[u].push_back(nu);
        }
      }
      vis[u] = true;
    }
  }

 protected:
  Query* query_;

  size_t h_part_num_;
  size_t dev_num_;
  size_t d_part_num_;

  ExecuteMode execute_mode_;
  Variant variant_;
};

#endif
