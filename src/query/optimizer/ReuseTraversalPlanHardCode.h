#ifndef __QUERY_OPTIMIZER_REUSE_TRAVERSAL_PLAN_HARD_CODE_H__
#define __QUERY_OPTIMIZER_REUSE_TRAVERSAL_PLAN_HARD_CODE_H__

#include "Plan.h"
#include "ReuseTraversalCommon.h"

class ReuseTraversalPlanHardcode {
 public:
  static void Generate(SearchSequence& seq,
                       LevelReuseIntersectPlan& level_reuse_intersect_plan,
                       const Query* query) {
    switch (query->GetQueryType()) {
      case Q0:
        Q0ReusePlan(seq, level_reuse_intersect_plan, query->GetVertexCount());
        break;
      case Q2:
        Q2ReusePlan(seq, level_reuse_intersect_plan, query->GetVertexCount());
        break;
      case Q3:
        Q3ReusePlan(seq, level_reuse_intersect_plan, query->GetVertexCount());
        break;
      case Q5:
        // Q5ReusePlan(seq, level_reuse_intersect_plan,
        // query->GetVertexCount());
        Q5ReusePlanAuto(seq, level_reuse_intersect_plan,
                        query->GetVertexCount());
        break;
      case Q6:
        Q6ReusePlan(seq, level_reuse_intersect_plan, query->GetVertexCount());
        break;
      case Q7:
        Q7ReusePlan(seq, level_reuse_intersect_plan, query->GetVertexCount());
        break;
      case Q8:
        Q8ReusePlan(seq, level_reuse_intersect_plan, query->GetVertexCount());
        // Q8ReusePlanAuto(seq, level_reuse_intersect_plan,
        // query->GetVertexCount());
        break;
      case Q9:
        Q9ReusePlan(seq, level_reuse_intersect_plan, query->GetVertexCount());
        // Q9ReusePlanAuto(seq, level_reuse_intersect_plan,
        // query->GetVertexCount()); Q9ReusePlanCase1(seq,
        // level_reuse_intersect_plan, query->GetVertexCount());
        break;
      case Q10:
        Q10ReusePlan(seq, level_reuse_intersect_plan, query->GetVertexCount());
        // Q10ReusePlanAuto(seq, level_reuse_intersect_plan,
        // query->GetVertexCount()); Q10ReusePlanCase1(seq,
        // level_reuse_intersect_plan, query->GetVertexCount());
        break;
      case Q11:
        Q11ReusePlan(seq, level_reuse_intersect_plan, query->GetVertexCount());
        break;
      case Q12:
        // Q12ReusePlan(seq, level_reuse_intersect_plan,
        // query->GetVertexCount());
        Q12ReusePlanAuto(seq, level_reuse_intersect_plan,
                         query->GetVertexCount());
        // Q12ReusePlanCase1(seq, level_reuse_intersect_plan,
        // query->GetVertexCount());
        break;
      case Q13:
        // Q13ReusePlan(seq, level_reuse_intersect_plan,
        // query->GetVertexCount());
        Q13ReusePlanAuto(seq, level_reuse_intersect_plan,
                         query->GetVertexCount());
        break;
      default:
        assert(false);
        break;
    }
  }

  static void Q0ReusePlan(SearchSequence& seq,
                          LevelReuseIntersectPlan& level_reuse_intersect_plan,
                          const size_t vertex_count) {
    for (size_t i = 0; i < vertex_count; ++i) seq[i] = i;
    level_reuse_intersect_plan.resize(vertex_count);
    level_reuse_intersect_plan[1].separate_conn_.push_back(0);
    level_reuse_intersect_plan[2].separate_conn_.push_back(0);
    level_reuse_intersect_plan[2].separate_conn_.push_back(1);
  }
  //// some hardcode plans for testing /////
  // /// note that the connectivity in the hardcode plan is based on
  // the index instead of the original vertex ids
  static void Q2ReusePlan(SearchSequence& seq,
                   LevelReuseIntersectPlan& level_reuse_intersect_plan, const
                       size_t vertex_count) {
    seq[0] = 0;
    seq[1] = 2;
    seq[2] = 1;
    seq[3] = 3;

    // 0 --- 3
    // | \   |
    // |   \ |
    // 2 --- 1

    // In the following, the pattern vertices are based on index
    level_reuse_intersect_plan.resize(vertex_count);
    level_reuse_intersect_plan[1].separate_conn_.push_back(0);
    level_reuse_intersect_plan[2].separate_conn_.push_back(0);
    level_reuse_intersect_plan[2].separate_conn_.push_back(1);
    {
      const uintV conn[] = {0, 1};
      const uintV source_conn[] = {0, 1};
      const uintV mapping[3][2] = {{0, 0}, {1, 1}, {2, 3}};
      ReuseConnMeta reuse_conn_meta(conn, 2, source_conn, 2, mapping, 3, 4);
      level_reuse_intersect_plan[3].reuse_conn_meta_.push_back(reuse_conn_meta);
    }
  }
  static void Q3ReusePlan(SearchSequence& seq,
                   LevelReuseIntersectPlan& level_reuse_intersect_plan, const
                       size_t vertex_count) {
    for (size_t i = 0; i < vertex_count; ++i) seq[i] = i;
    // 0 --- 1
    // | \ / |
    // | / \ |
    // 2 --- 3
    level_reuse_intersect_plan.resize(vertex_count);
    level_reuse_intersect_plan[1].separate_conn_.push_back(0);
    level_reuse_intersect_plan[2].separate_conn_.push_back(0);
    level_reuse_intersect_plan[2].separate_conn_.push_back(1);
    {
      const uintV conn[] = {0, 1};
      const uintV source_conn[] = {0, 1};
      const uintV mapping[3][2] = {{0, 0}, {1, 1}, {2, 3}};
      ReuseConnMeta reuse_conn_meta(conn, 2, source_conn, 2, mapping, 3, 4);
      level_reuse_intersect_plan[3].reuse_conn_meta_.push_back(reuse_conn_meta);
      level_reuse_intersect_plan[3].separate_conn_.push_back(2);
    }
  }

  static void Q5ReusePlan(SearchSequence& seq,
                   LevelReuseIntersectPlan& level_reuse_intersect_plan, const
                       size_t vertex_count) {
    // four triangle
    // 4---0---5
    // | / | \ |
    // 2---1---3
    //
    seq[0] = 0;
    seq[1] = 3;
    seq[2] = 2;
    seq[3] = 4;
    seq[4] = 1;
    seq[5] = 5;

    level_reuse_intersect_plan.resize(vertex_count);
    level_reuse_intersect_plan[1].separate_conn_.push_back(0);
    level_reuse_intersect_plan[2].separate_conn_.push_back(0);
    level_reuse_intersect_plan[2].separate_conn_.push_back(1);
    {
      const uintV conn[] = {0, 1};
      const uintV source_conn[] = {0, 1};
      const uintV mapping[3][2] = {{0, 0}, {1, 1}, {2, 3}};
      ReuseConnMeta reuse_conn_meta(conn, 2, source_conn, 2, mapping, 3, 6);
      ;
      level_reuse_intersect_plan[3].reuse_conn_meta_.push_back(reuse_conn_meta);
    }
    {
      const uintV conn[] = {0, 2};
      const uintV source_conn[] = {0, 1};
      const uintV mapping[3][2] = {{0, 0}, {1, 2}, {2, 4}};
      ReuseConnMeta reuse_conn_meta(conn, 2, source_conn, 2, mapping, 3, 6);
      level_reuse_intersect_plan[4].reuse_conn_meta_.push_back(reuse_conn_meta);
    }
    {
      const uintV conn[] = {0, 3};
      const uintV source_conn[] = {0, 1};
      const uintV mapping[3][2] = {{0, 0}, {1, 3}, {3, 5}};
      ReuseConnMeta reuse_conn_meta(conn, 3, source_conn, 2, mapping, 3, 6);
      ;
      level_reuse_intersect_plan[5].reuse_conn_meta_.push_back(reuse_conn_meta);
    }
  }

static   void Q5ReusePlanAuto(SearchSequence& seq,
                       LevelReuseIntersectPlan& level_reuse_intersect_plan, const
                           size_t vertex_count) {
    // four triangle
    // 4---0---5
    // | / | \ |
    // 2---1---3
    //
    seq[0] = 0;
    seq[1] = 2;
    seq[2] = 3;
    seq[3] = 4;
    seq[4] = 5;
    seq[5] = 1;

    level_reuse_intersect_plan.resize(vertex_count);
    level_reuse_intersect_plan[1].separate_conn_.push_back(0);
    level_reuse_intersect_plan[2].separate_conn_.push_back(0);
    level_reuse_intersect_plan[2].separate_conn_.push_back(1);
    {
      const uintV conn[] = {0, 2};
      const uintV source_conn[] = {0, 1};
      const uintV mapping[3][2] = {{0, 0}, {1, 2}, {2, 3}};
      ReuseConnMeta reuse_conn_meta(conn, 2, source_conn, 2, mapping, 3, 6);
      level_reuse_intersect_plan[3].reuse_conn_meta_.push_back(reuse_conn_meta);
    }
    {
      const uintV conn[] = {0, 3};
      const uintV source_conn[] = {0, 1};
      const uintV mapping[3][2] = {{0, 0}, {1, 3}, {2, 4}};
      ReuseConnMeta reuse_conn_meta(conn, 2, source_conn, 2, mapping, 3, 6);
      level_reuse_intersect_plan[4].reuse_conn_meta_.push_back(reuse_conn_meta);
    }
    {
      const uintV conn[] = {0, 1};
      const uintV source_conn[] = {0, 1};
      const uintV mapping[3][2] = {{0, 0}, {1, 1}, {2, 5}};
      ReuseConnMeta reuse_conn_meta(conn, 2, source_conn, 2, mapping, 3, 6);
      level_reuse_intersect_plan[5].reuse_conn_meta_.push_back(reuse_conn_meta);
    }
  }

  static void Q6ReusePlan(SearchSequence& seq,
                   LevelReuseIntersectPlan& level_reuse_intersect_plan, const
                       size_t vertex_count) {
    seq[0] = 0;
    seq[1] = 3;
    seq[2] = 1;
    seq[3] = 2;
    seq[4] = 4;

    //    4
    //   /  \
    //  0----1
    //  | \/ |
    //  | /\ |
    //  2----3

    level_reuse_intersect_plan.resize(vertex_count);
    level_reuse_intersect_plan[1].separate_conn_.push_back(0);
    level_reuse_intersect_plan[2].separate_conn_.push_back(0);
    level_reuse_intersect_plan[2].separate_conn_.push_back(1);
    {
      const uintV conn[] = {0, 1};
      const uintV source_conn[] = {0, 1};
      const uintV mapping[3][2] = {{0, 0}, {1, 1}, {2, 3}};
      ReuseConnMeta reuse_conn_meta(conn, 2, source_conn, 2, mapping, 3, 5);
      level_reuse_intersect_plan[3].reuse_conn_meta_.push_back(reuse_conn_meta);
      level_reuse_intersect_plan[3].separate_conn_.push_back(2);
    }
    {
      const uintV conn[] = {0, 1};
      const uintV source_conn[] = {0, 1};
      const uintV mapping[3][2] = {{0, 0}, {1, 1}, {2, 4}};
      ReuseConnMeta reuse_conn_meta(conn, 2, source_conn, 2, mapping, 3, 5);
      level_reuse_intersect_plan[4].reuse_conn_meta_.push_back(reuse_conn_meta);
    }
  }

  static void Q7ReusePlan(SearchSequence& seq,
                   LevelReuseIntersectPlan& level_reuse_intersect_plan, const
                       size_t vertex_count) {
    for (size_t i = 0; i < vertex_count; ++i) seq[i] = i;
    // 5 clique
    level_reuse_intersect_plan.resize(vertex_count);
    level_reuse_intersect_plan[1].separate_conn_.push_back(0);
    level_reuse_intersect_plan[2].separate_conn_.push_back(0);
    level_reuse_intersect_plan[2].separate_conn_.push_back(1);
    {
      const uintV conn[] = {0, 1};
      const uintV source_conn[] = {0, 1};
      const uintV mapping[3][2] = {{0, 0}, {1, 1}, {2, 3}};
      ReuseConnMeta reuse_conn_meta(conn, 2, source_conn, 2, mapping, 3, 5);
      level_reuse_intersect_plan[3].reuse_conn_meta_.push_back(reuse_conn_meta);
      level_reuse_intersect_plan[3].separate_conn_.push_back(2);
    }
    {
      const uintV conn[] = {0, 1, 2};
      const uintV source_conn[] = {0, 1, 2};
      const uintV mapping[4][2] = {{0, 0}, {1, 1}, {2, 2}, {3, 4}};
      ReuseConnMeta reuse_conn_meta(conn, 3, source_conn, 3, mapping, 4, 5);
      level_reuse_intersect_plan[4].reuse_conn_meta_.push_back(reuse_conn_meta);
      level_reuse_intersect_plan[4].separate_conn_.push_back(3);
    }
  }

  static void Q8ReusePlan(SearchSequence& seq,
                   LevelReuseIntersectPlan& level_reuse_intersect_plan, const
                       size_t vertex_count) {
    for (size_t i = 0; i < vertex_count; ++i) seq[i] = i;
    //    4
    //  /   \
    // 3     2
    // |\   /|
    // | \ / |
    // | / \ |
    // |/   \|
    // 0 --- 1
    //
    level_reuse_intersect_plan.resize(vertex_count);
    level_reuse_intersect_plan[1].separate_conn_.push_back(0);
    level_reuse_intersect_plan[2].separate_conn_.push_back(0);
    level_reuse_intersect_plan[2].separate_conn_.push_back(1);
    {
      const uintV conn[] = {0, 1};
      const uintV source_conn[] = {0, 1};
      const uintV mapping[3][2] = {{0, 0}, {1, 1}, {2, 3}};
      ReuseConnMeta reuse_conn_meta(conn, 2, source_conn, 2, mapping, 3, 5);
      level_reuse_intersect_plan[3].reuse_conn_meta_.push_back(reuse_conn_meta);
    }
    level_reuse_intersect_plan[4].separate_conn_.push_back(2);
    level_reuse_intersect_plan[4].separate_conn_.push_back(3);
  }

  static void Q8ReusePlanAuto(SearchSequence& seq,
                       LevelReuseIntersectPlan& level_reuse_intersect_plan, const
                           size_t vertex_count) {
    seq[0] = 0;
    seq[1] = 1;
    seq[2] = 2;
    seq[3] = 4;
    seq[4] = 3;

    //    3
    //  /   \
    // 4     2
    // |\   /|
    // | \ / |
    // | / \ |
    // |/   \|
    // 0 --- 1
    //
    level_reuse_intersect_plan.resize(vertex_count);
    level_reuse_intersect_plan[1].separate_conn_.push_back(0);
    level_reuse_intersect_plan[2].separate_conn_.push_back(0);
    level_reuse_intersect_plan[2].separate_conn_.push_back(1);
    level_reuse_intersect_plan[3].separate_conn_.push_back(2);

    {
      const uintV conn[] = {0, 1};
      const uintV source_conn[] = {0, 1};
      const uintV mapping[3][2] = {{0, 0}, {1, 1}, {2, 4}};
      ReuseConnMeta reuse_conn_meta(conn, 2, source_conn, 2, mapping, 3, 5);
      level_reuse_intersect_plan[4].reuse_conn_meta_.push_back(reuse_conn_meta);
      level_reuse_intersect_plan[4].separate_conn_.push_back(3);
    }
  }

  static void Q9ReusePlan(SearchSequence& seq,
                   LevelReuseIntersectPlan& level_reuse_intersect_plan, const
                       size_t vertex_count) {
    for (size_t i = 0; i < vertex_count; ++i) seq[i] = i;
    // three triangle
    //    0
    //  / /\ \
    // 3 /  \ 4
    // |/    \|
    // 1------2
    //
    level_reuse_intersect_plan.resize(vertex_count);
    level_reuse_intersect_plan[1].separate_conn_.push_back(0);
    level_reuse_intersect_plan[2].separate_conn_.push_back(0);
    level_reuse_intersect_plan[2].separate_conn_.push_back(1);
    {
      const uintV conn[] = {0, 1};
      const uintV source_conn[] = {0, 1};
      const uintV mapping[3][2] = {{0, 0}, {1, 1}, {2, 3}};
      ReuseConnMeta reuse_conn_meta(conn, 2, source_conn, 2, mapping, 3, 5);
      level_reuse_intersect_plan[3].reuse_conn_meta_.push_back(reuse_conn_meta);
    }
    {
      const uintV conn[] = {0, 2};
      const uintV source_conn[] = {0, 1};
      const uintV mapping[3][2] = {{0, 0}, {1, 2}, {2, 4}};
      ReuseConnMeta reuse_conn_meta(conn, 2, source_conn, 2, mapping, 3, 5);
      level_reuse_intersect_plan[4].reuse_conn_meta_.push_back(reuse_conn_meta);
    }
  }

  static void Q9ReusePlanAuto(SearchSequence& seq,
                       LevelReuseIntersectPlan& level_reuse_intersect_plan, const
                           size_t vertex_count) {
    seq[0] = 0;
    seq[1] = 1;
    seq[2] = 2;
    seq[3] = 4;
    seq[4] = 3;

    // three triangle
    //    0
    //  / /\ \
    // 4 /  \ 3
    // |/    \|
    // 1------2
    //
    level_reuse_intersect_plan.resize(vertex_count);
    level_reuse_intersect_plan[1].separate_conn_.push_back(0);
    level_reuse_intersect_plan[2].separate_conn_.push_back(0);
    level_reuse_intersect_plan[2].separate_conn_.push_back(1);
    {
      const uintV conn[] = {0, 2};
      const uintV source_conn[] = {0, 1};
      const uintV mapping[3][2] = {{0, 0}, {1, 2}, {2, 3}};
      ReuseConnMeta reuse_conn_meta(conn, 2, source_conn, 2, mapping, 3, 5);
      level_reuse_intersect_plan[3].reuse_conn_meta_.push_back(reuse_conn_meta);
    }
    {
      const uintV conn[] = {0, 1};
      const uintV source_conn[] = {0, 1};
      const uintV mapping[3][2] = {{0, 0}, {1, 1}, {2, 4}};
      ReuseConnMeta reuse_conn_meta(conn, 2, source_conn, 2, mapping, 3, 5);
      level_reuse_intersect_plan[4].reuse_conn_meta_.push_back(reuse_conn_meta);
    }
  }

  static void Q9ReusePlanCase1(
      SearchSequence& seq,
      LevelReuseIntersectPlan& level_reuse_intersect_plan, const size_t
          vertex_count) {
    for (size_t i = 0; i < vertex_count; ++i) seq[i] = i;
    // This is similar to Q9ReusePlan, but disable the reuse for u4.
    level_reuse_intersect_plan.resize(vertex_count);
    level_reuse_intersect_plan[1].separate_conn_.push_back(0);
    level_reuse_intersect_plan[2].separate_conn_.push_back(0);
    level_reuse_intersect_plan[2].separate_conn_.push_back(1);
    {
      const uintV conn[] = {0, 1};
      const uintV source_conn[] = {0, 1};
      const uintV mapping[3][2] = {{0, 0}, {1, 1}, {2, 3}};
      ReuseConnMeta reuse_conn_meta(conn, 2, source_conn, 2, mapping, 3, 5);
      level_reuse_intersect_plan[3].reuse_conn_meta_.push_back(reuse_conn_meta);
    }
    level_reuse_intersect_plan[4].separate_conn_.push_back(0);
    level_reuse_intersect_plan[4].separate_conn_.push_back(2);
  }

  static void Q10ReusePlan(SearchSequence& seq,
                    LevelReuseIntersectPlan& level_reuse_intersect_plan, const
                        size_t vertex_count) {
    // 3 --- 4 --- 2
    //  \ \  |   / |
    //   \ --|---  |
    //    \  | /  \|
    //     \ 0 --- 1
    //
    seq[0] = 0;
    seq[1] = 1;
    seq[2] = 2;
    seq[3] = 4;
    seq[4] = 3;

    level_reuse_intersect_plan.resize(vertex_count);
    level_reuse_intersect_plan[1].separate_conn_.push_back(0);
    level_reuse_intersect_plan[2].separate_conn_.push_back(0);
    level_reuse_intersect_plan[2].separate_conn_.push_back(1);
    {
      const uintV conn[] = {0, 1};
      const uintV source_conn[] = {0, 1};
      const uintV mapping[3][2] = {{0, 0}, {1, 1}, {2, 3}};
      ReuseConnMeta reuse_conn_meta(conn, 2, source_conn, 2, mapping, 3, 5);
      level_reuse_intersect_plan[3].reuse_conn_meta_.push_back(reuse_conn_meta);
    }
    {
      const uintV conn[] = {0, 3};
      const uintV source_conn[] = {0, 1};
      const uintV mapping[3][2] = {{0, 0}, {1, 3}, {2, 4}};
      ReuseConnMeta reuse_conn_meta(conn, 2, source_conn, 2, mapping, 3, 5);
      level_reuse_intersect_plan[4].reuse_conn_meta_.push_back(reuse_conn_meta);
      level_reuse_intersect_plan[4].separate_conn_.push_back(2);
    }
  }

  static void Q10ReusePlanAuto(
      SearchSequence& seq,
      LevelReuseIntersectPlan& level_reuse_intersect_plan, const size_t
          vertex_count) {
    for (size_t i = 0; i < vertex_count; ++i) seq[i] = i;
    level_reuse_intersect_plan.resize(vertex_count);
    level_reuse_intersect_plan[1].separate_conn_.push_back(0);
    level_reuse_intersect_plan[2].separate_conn_.push_back(0);
    level_reuse_intersect_plan[2].separate_conn_.push_back(1);
    {
      const uintV conn[] = {0, 2};
      const uintV source_conn[] = {0, 1};
      const uintV mapping[3][2] = {{0, 0}, {1, 2}, {2, 3}};
      ReuseConnMeta reuse_conn_meta(conn, 2, source_conn, 2, mapping, 3, 5);
      level_reuse_intersect_plan[3].reuse_conn_meta_.push_back(reuse_conn_meta);
    }
    {
      const uintV conn[] = {0, 1};
      const uintV source_conn[] = {0, 1};
      const uintV mapping[3][2] = {{0, 0}, {1, 1}, {2, 4}};
      ReuseConnMeta reuse_conn_meta(conn, 2, source_conn, 2, mapping, 3, 5);
      level_reuse_intersect_plan[4].reuse_conn_meta_.push_back(reuse_conn_meta);
      level_reuse_intersect_plan[4].separate_conn_.push_back(3);
    }
  }

  static void Q10ReusePlanCase1(
      SearchSequence& seq,
      LevelReuseIntersectPlan& level_reuse_intersect_plan, const size_t
          vertex_count) {
    // This is similar to Q10ReusePlan, except that disabling the reuse for u4
    seq[0] = 0;
    seq[1] = 1;
    seq[2] = 2;
    seq[3] = 4;
    seq[4] = 3;

    level_reuse_intersect_plan.resize(vertex_count);
    level_reuse_intersect_plan[1].separate_conn_.push_back(0);
    level_reuse_intersect_plan[2].separate_conn_.push_back(0);
    level_reuse_intersect_plan[2].separate_conn_.push_back(1);
    {
      const uintV conn[] = {0, 1};
      const uintV source_conn[] = {0, 1};
      const uintV mapping[3][2] = {{0, 0}, {1, 1}, {2, 3}};
      ReuseConnMeta reuse_conn_meta(conn, 2, source_conn, 2, mapping, 3, 5);
      level_reuse_intersect_plan[3].reuse_conn_meta_.push_back(reuse_conn_meta);
    }
    level_reuse_intersect_plan[4].separate_conn_.push_back(0);
    level_reuse_intersect_plan[4].separate_conn_.push_back(2);
    level_reuse_intersect_plan[4].separate_conn_.push_back(3);
  }

  static void Q11ReusePlan(SearchSequence& seq,
                    LevelReuseIntersectPlan& level_reuse_intersect_plan, const
                        size_t vertex_count) {
    for (size_t i = 0; i < vertex_count; ++i) seq[i] = i;
    // 6 clique
    level_reuse_intersect_plan.resize(vertex_count);
    level_reuse_intersect_plan[1].separate_conn_.push_back(0);
    level_reuse_intersect_plan[2].separate_conn_.push_back(0);
    level_reuse_intersect_plan[2].separate_conn_.push_back(1);
    {
      const uintV conn[] = {0, 1};
      const uintV source_conn[] = {0, 1};
      const uintV mapping[3][2] = {{0, 0}, {1, 1}, {2, 3}};
      ReuseConnMeta reuse_conn_meta(conn, 2, source_conn, 2, mapping, 3, 6);
      level_reuse_intersect_plan[3].reuse_conn_meta_.push_back(reuse_conn_meta);
      level_reuse_intersect_plan[3].separate_conn_.push_back(2);
    }
    {
      const uintV conn[] = {0, 1, 2};
      const uintV source_conn[] = {0, 1, 2};
      const uintV mapping[4][2] = {{0, 0}, {1, 1}, {2, 2}, {3, 4}};
      ReuseConnMeta reuse_conn_meta(conn, 3, source_conn, 3, mapping, 4, 6);
      level_reuse_intersect_plan[4].reuse_conn_meta_.push_back(reuse_conn_meta);
      level_reuse_intersect_plan[4].separate_conn_.push_back(3);
    }
    {
      const uintV conn[] = {0, 1, 2, 3};
      const uintV source_conn[] = {0, 1, 2, 3};
      const uintV mapping[5][2] = {{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 5}};
      ReuseConnMeta reuse_conn_meta(conn, 4, source_conn, 4, mapping, 5, 6);
      level_reuse_intersect_plan[5].reuse_conn_meta_.push_back(reuse_conn_meta);
      level_reuse_intersect_plan[5].separate_conn_.push_back(4);
    }
  }

  static void Q12ReusePlan(SearchSequence& seq,
                    LevelReuseIntersectPlan& level_reuse_intersect_plan, const
                        size_t vertex_count) {
    for (size_t i = 0; i < vertex_count; ++i) seq[i] = i;
    level_reuse_intersect_plan.resize(vertex_count);
    level_reuse_intersect_plan[1].separate_conn_.push_back(0);
    level_reuse_intersect_plan[2].separate_conn_.push_back(0);
    level_reuse_intersect_plan[2].separate_conn_.push_back(1);
    {
      const uintV conn[] = {0, 1};
      const uintV source_conn[] = {0, 1};
      const uintV mapping[3][2] = {{0, 0}, {1, 1}, {2, 3}};
      ReuseConnMeta reuse_conn_meta(conn, 2, source_conn, 2, mapping, 3, 5);
      level_reuse_intersect_plan[3].reuse_conn_meta_.push_back(reuse_conn_meta);
      level_reuse_intersect_plan[3].separate_conn_.push_back(2);
    }
    {
      const uintV conn[] = {0, 1, 3};
      const uintV source_conn[] = {0, 1, 2};
      const uintV mapping[4][2] = {{0, 0}, {1, 1}, {2, 3}, {3, 4}};
      ReuseConnMeta reuse_conn_meta(conn, 3, source_conn, 3, mapping, 4, 5);
      level_reuse_intersect_plan[4].reuse_conn_meta_.push_back(reuse_conn_meta);
    }
  }

  static void Q12ReusePlanAuto(
      SearchSequence& seq,
      LevelReuseIntersectPlan& level_reuse_intersect_plan, const size_t
          vertex_count) {
    seq[0] = 0;
    seq[1] = 1;
    seq[2] = 3;
    seq[3] = 2;
    seq[4] = 4;

    level_reuse_intersect_plan.resize(vertex_count);
    level_reuse_intersect_plan[1].separate_conn_.push_back(0);
    level_reuse_intersect_plan[2].separate_conn_.push_back(0);
    level_reuse_intersect_plan[2].separate_conn_.push_back(1);
    {
      const uintV conn[] = {0, 1};
      const uintV source_conn[] = {0, 1};
      const uintV mapping[3][2] = {{0, 0}, {1, 1}, {2, 3}};
      ReuseConnMeta reuse_conn_meta(conn, 2, source_conn, 2, mapping, 3, 5);
      level_reuse_intersect_plan[3].reuse_conn_meta_.push_back(reuse_conn_meta);
      level_reuse_intersect_plan[3].separate_conn_.push_back(2);
    }
    {
      const uintV conn[] = {0, 1, 2};
      const uintV source_conn[] = {0, 1, 2};
      const uintV mapping[4][2] = {{0, 0}, {1, 1}, {2, 2}, {3, 4}};
      ReuseConnMeta reuse_conn_meta(conn, 3, source_conn, 3, mapping, 4, 5);
      level_reuse_intersect_plan[4].reuse_conn_meta_.push_back(reuse_conn_meta);
    }
  }

  static void Q12ReusePlanCase1(
      SearchSequence& seq,
      LevelReuseIntersectPlan& level_reuse_intersect_plan, const size_t
          vertex_count) {
    for (size_t i = 0; i < vertex_count; ++i) seq[i] = i;
    // This is similar to Q12ReusePlan, except that disabling the reuse for u4
    level_reuse_intersect_plan.resize(vertex_count);
    level_reuse_intersect_plan[1].separate_conn_.push_back(0);
    level_reuse_intersect_plan[2].separate_conn_.push_back(0);
    level_reuse_intersect_plan[2].separate_conn_.push_back(1);
    {
      const uintV conn[] = {0, 1};
      const uintV source_conn[] = {0, 1};
      const uintV mapping[3][2] = {{0, 0}, {1, 1}, {2, 3}};
      ReuseConnMeta reuse_conn_meta(conn, 2, source_conn, 2, mapping, 3, 5);
      level_reuse_intersect_plan[3].reuse_conn_meta_.push_back(reuse_conn_meta);
      level_reuse_intersect_plan[3].separate_conn_.push_back(2);
    }
    level_reuse_intersect_plan[4].separate_conn_.push_back(0);
    level_reuse_intersect_plan[4].separate_conn_.push_back(1);
    level_reuse_intersect_plan[4].separate_conn_.push_back(3);
  }

  static void Q13ReusePlan(SearchSequence& seq,
                    LevelReuseIntersectPlan& level_reuse_intersect_plan, const
                        size_t vertex_count) {
    seq[0] = 0;
    seq[1] = 1;
    seq[2] = 2;
    seq[3] = 4;
    seq[4] = 3;
    seq[5] = 5;

    level_reuse_intersect_plan.resize(vertex_count);
    level_reuse_intersect_plan[1].separate_conn_.push_back(0);
    level_reuse_intersect_plan[2].separate_conn_.push_back(0);
    level_reuse_intersect_plan[2].separate_conn_.push_back(1);
    level_reuse_intersect_plan[3].separate_conn_.push_back(1);
    level_reuse_intersect_plan[3].separate_conn_.push_back(2);
    {
      const uintV conn[] = {0, 1};
      const uintV source_conn[] = {0, 1};
      const uintV mapping[3][2] = {{0, 0}, {1, 1}, {2, 4}};
      ReuseConnMeta reuse_conn_meta(conn, 2, source_conn, 2, mapping, 3, 6);
      level_reuse_intersect_plan[4].reuse_conn_meta_.push_back(reuse_conn_meta);
    }
    {
      const uintV conn[] = {0, 2};
      const uintV source_conn[] = {0, 1};
      const uintV mapping[3][2] = {{0, 0}, {1, 2}, {2, 5}};
      ReuseConnMeta reuse_conn_meta(conn, 2, source_conn, 2, mapping, 3, 6);
      level_reuse_intersect_plan[5].reuse_conn_meta_.push_back(reuse_conn_meta);
    }
  }

  static void Q13ReusePlanAuto(SearchSequence& seq,
                        LevelReuseIntersectPlan& level_reuse_intersect_plan,
                        const size_t vertex_count) {
    seq[0] = 0;
    seq[1] = 1;
    seq[2] = 2;
    seq[3] = 5;
    seq[4] = 4;
    seq[5] = 3;

    level_reuse_intersect_plan.resize(vertex_count);
    level_reuse_intersect_plan[1].separate_conn_.push_back(0);
    level_reuse_intersect_plan[2].separate_conn_.push_back(0);
    level_reuse_intersect_plan[2].separate_conn_.push_back(1);
    {
      const uintV conn[] = {0, 2};
      const uintV source_conn[] = {0, 1};
      const uintV mapping[3][2] = {{0, 0}, {1, 2}, {2, 3}};
      ReuseConnMeta reuse_conn_meta(conn, 2, source_conn, 2, mapping, 3, 6);
      level_reuse_intersect_plan[3].reuse_conn_meta_.push_back(reuse_conn_meta);
    }

    {
      const uintV conn[] = {1, 2};
      const uintV source_conn[] = {0, 1};
      const uintV mapping[3][2] = {{0, 1}, {1, 2}, {2, 4}};
      ReuseConnMeta reuse_conn_meta(conn, 2, source_conn, 2, mapping, 3, 6);
      level_reuse_intersect_plan[4].reuse_conn_meta_.push_back(reuse_conn_meta);
    }
    {
      const uintV conn[] = {0, 1};
      const uintV source_conn[] = {0, 1};
      const uintV mapping[3][2] = {{0, 0}, {1, 1}, {2, 5}};
      ReuseConnMeta reuse_conn_meta(conn, 2, source_conn, 2, mapping, 3, 6);
      level_reuse_intersect_plan[5].reuse_conn_meta_.push_back(reuse_conn_meta);
    }
  }
};
#endif