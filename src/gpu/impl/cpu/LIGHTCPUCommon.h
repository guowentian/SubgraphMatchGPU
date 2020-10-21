#ifndef __HYBRID_LIGHT_PARTIAL_GROUP_CPU_COMPONENT_COMMON_H__
#define __HYBRID_LIGHT_PARTIAL_GROUP_CPU_COMPONENT_COMMON_H__

#include "HybridPtGroupCPUComp.h"
#include "LazyTraversalPlan.h"

namespace LightCPU {
template <typename T>
using Array = std::vector<T>;

template <typename T>
using LayeredArray = std::vector<Array<T> >;

struct LightCPUWorkContext {
  size_t group_id;
  size_t dfs_id;
  long long *ans;

  // intermediate result
  Array<uintV> path;
  LayeredArray<uintV> candidates;

  void Init(size_t group_id, size_t dfs_id, long long *ans,
            size_t vertex_count) {
    this->group_id = group_id;
    this->dfs_id = dfs_id;
    this->ans = ans;
    this->path.resize(vertex_count);
    this->candidates.clear();
    this->candidates.resize(vertex_count);
  }
};

struct PlanMeta {
  // backward connectivity for COMPUTE
  AllConnType conn;
  // the conditions among all vertices, exclude NON_EQUAL
  AllCondType cond;
  // For each exec level, the condition with the materialized
  // vertices
  AllCondType materialized_cond;
  // During COUNT or COMPUTE_COUNT, the conditions to all materialized vertices
  AllCondType count_cond;

  // For each exec level, the set of materialized and unmaterialized vertices
  MultiVTGroup materialized_vertices;
  MultiVTGroup computed_unmaterialized_vertices;

  void Init(const AllConnType &backward_conn, const AllCondType &cond,
            const MultiVTGroup &materialized_vertices,
            const MultiVTGroup &computed_unmaterialized_vertices,
            const std::vector<LazyTraversalEntry> &exec_seq) {
    size_t c = backward_conn.size();
    size_t l = materialized_vertices.size();

    this->conn.resize(c);
    this->cond.resize(c);
    for (size_t i = 0; i < c; ++i) {
      this->conn[i].assign(backward_conn[i].begin(), backward_conn[i].end());
      this->cond[i].assign(cond[i].begin(), cond[i].end());
    }

    this->materialized_cond.resize(l);
    for (size_t i = 0; i < l; ++i) {
      if (exec_seq[i].first != COUNT) {
        uintV cur_u = exec_seq[i].second;
        for (auto u : materialized_vertices[i]) {
          auto cond_op = Plan::GetConditionType(cur_u, u, cond);
          this->materialized_cond[i].push_back(std::make_pair(cond_op, u));
        }
      }
    }

    this->count_cond.resize(c);
    for (size_t i = 0; i < l; ++i) {
      if (exec_seq[i].first == COMPUTE_COUNT || exec_seq[i].first == COUNT) {
        // the condition among materialized vertices
        for (auto u1 : materialized_vertices[i]) {
          for (auto u2 : materialized_vertices[i]) {
            if (u1 != u2) {
              auto cond_op = Plan::GetConditionType(u1, u2, cond);
              this->count_cond[u1].push_back(std::make_pair(cond_op, u2));
            }
          }
        }

        // the condition between unmaterialized and materialized vertices
        VTGroup comp_umvs(computed_unmaterialized_vertices[i]);
        if (exec_seq[i].first == COMPUTE_COUNT) {
          comp_umvs.push_back(exec_seq[i].second);
        }
        for (auto u1 : comp_umvs) {
          for (auto u2 : materialized_vertices[i]) {
            assert(u1 != u2);
            auto cond_op = Plan::GetConditionType(u1, u2, cond);
            this->count_cond[u1].push_back(std::make_pair(cond_op, u2));
          }
        }
      }
    }

    this->materialized_vertices.resize(l);
    this->computed_unmaterialized_vertices.resize(l);
    for (size_t i = 0; i < l; ++i) {
      this->materialized_vertices[i].assign(materialized_vertices[i].begin(),
                                            materialized_vertices[i].end());
      this->computed_unmaterialized_vertices[i].assign(
          computed_unmaterialized_vertices[i].begin(),
          computed_unmaterialized_vertices[i].end());
    }
  }

  void Print() const {
    size_t vertex_count = conn.size();
    for (uintV u = 0; u < vertex_count; ++u) {
      std::cout << "u=" << u << ", connectivity:";
      for (auto u2 : conn[u]) std::cout << " " << u2;
      std::cout << std::endl;

      std::cout << "condition:";
      for (auto entry : cond[u]) {
        std::cout << " (" << u << " " << GetString(entry.first) << " "
                  << entry.second << ")";
      }
      std::cout << std::endl;

      std::cout << "count condition:";
      for (auto entry : count_cond[u]) {
        std::cout << " (" << u << " " << GetString(entry.first) << " "
                  << entry.second << ")";
      }
      std::cout << std::endl;
    }

    for (size_t l = 0; l < materialized_cond.size(); ++l) {
      std::cout << "level=" << l << ", materialized_cond:";
      for (auto entry : materialized_cond[l]) {
        std::cout << " (" << GetString(entry.first) << " " << entry.second
                  << ")";
      }
      std::cout << std::endl;

      std::cout << "materialized vertices:";
      for (auto u : materialized_vertices[l]) std::cout << " " << u;
      std::cout << std::endl;

      std::cout << "computed_unmaterialized_vertices:";
      for (auto u : computed_unmaterialized_vertices[l]) std::cout << " " << u;
      std::cout << std::endl;
    }
  }
};

struct WorkMeta {
  LazyTraversalPlan *plan;
  // The plan used to compute a group of dfs
  std::vector<PlanMeta> group_meta;
  // The plan for each specific dfs
  std::vector<PlanMeta> dfs_meta;

  void Init(LazyTraversalPlan *plan) {
    this->plan = plan;
    auto &group_dfs_ids = plan->GetGroupDfsIds();
    auto &dfs_orders = plan->GetSearchSequences();

    group_meta.resize(group_dfs_ids.size());
    dfs_meta.resize(dfs_orders.size());
    for (size_t group_id = 0; group_id < group_dfs_ids.size(); ++group_id) {
      size_t dfs_id = group_dfs_ids[group_id][0];

      AllConnType backward_conn;
      Plan::GetOrderedIndexBasedConnectivity(
          backward_conn, plan->GetConnectivity(), dfs_orders[dfs_id]);
      // To find instances for a group, simply ensure the non-equality
      AllCondType nil_cond(plan->GetVertexCount());
      // materialized_vertices and computed_unmaterialized_vertices in the plan
      // are index based
      auto &materialized_vertices =
          plan->GetInterPartitionMaterializedVertices()[dfs_id];
      auto &computed_unmaterialized_vertices =
          plan->GetInterPartitionComputedUnmaterializedVertices()[dfs_id];

      group_meta[group_id].Init(
          backward_conn, nil_cond, materialized_vertices,
          computed_unmaterialized_vertices,
          plan->GetInterPartitionExecuteOperations()[dfs_id]);

      for (auto dfs_id : group_dfs_ids[group_id]) {
        // use the ordering for a specific dfs
        AllCondType cond;
        Plan::GetIndexBasedOrdering(cond, plan->GetOrdering(),
                                    dfs_orders[dfs_id]);

        dfs_meta[dfs_id].Init(
            backward_conn, cond, materialized_vertices,
            computed_unmaterialized_vertices,
            plan->GetInterPartitionExecuteOperations()[dfs_id]);
      }
    }
  }
  void Print() const {
    for (size_t group_id = 0; group_id < group_meta.size(); ++group_id) {
      std::cout << "=========group_meta, group_id=" << group_id
                << " ==========" << std::endl;
      group_meta[group_id].Print();
    }

    for (size_t dfs_id = 0; dfs_id < dfs_meta.size(); ++dfs_id) {
      std::cout << "=========dfs_meta, dfs_id=" << dfs_id
                << "============" << std::endl;
      dfs_meta[dfs_id].Print();
    }
  }
};
}  // namespace LightCPU

#endif