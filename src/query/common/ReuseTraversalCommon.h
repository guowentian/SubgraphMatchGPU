#ifndef __QUERY_REUSE_TRAVERSAL_COMMON_H__
#define __QUERY_REUSE_TRAVERSAL_COMMON_H__

#include "QueryCommon.h"

#include <algorithm>
#include <cassert>
#include <vector>

// With n pattern vertex, a mapping from n to n vertices
typedef std::vector<uintV> VertexMapping;

// For a pattern vertex, a set of connectivity to be reused
// The data structure describe some reuse information
struct ReuseConnMeta {
  ReuseConnMeta() {}
  // most general constructor
  ReuseConnMeta(ConnType& conn, uintV source_vertex, ConnType& source_conn,
                VertexMapping& mapping, VertexMapping& inverted_mapping) {
    Init(conn, source_vertex, source_conn, mapping, inverted_mapping);
  }

  // easy-to use constructor
  ReuseConnMeta(const uintV* conn, uintV source_vertex,
                const uintV* source_conn, size_t conn_size,
                const uintV pairs[][2], size_t pairs_num, size_t vertex_count) {
    VertexMapping mapping(vertex_count, vertex_count);
    for (size_t i = 0; i < pairs_num; ++i) {
      mapping[pairs[i][0]] = pairs[i][1];
    }
    Init(conn, source_vertex, source_conn, conn_size, mapping);
  }

  void Init(const uintV* conn, uintV source_vertex, const uintV* source_conn,
            size_t conn_size, VertexMapping& mapping) {
    ConnType conn_vec(conn, conn + conn_size);
    ConnType source_conn_vec(source_conn, source_conn + conn_size);

    size_t vertex_count = mapping.size();
    VertexMapping inverted_mapping(vertex_count, vertex_count);
    for (uintV u = 0; u < vertex_count; ++u) {
      if (mapping[u] < vertex_count) {
        inverted_mapping[mapping[u]] = u;
      }
    }

    //#if defined(DEBUG)
    std::vector<bool> vis(vertex_count, false);
    for (uintV u = 0; u < vertex_count; ++u) {
      if (mapping[u] < vertex_count) {
        vis[u] = true;
      }
    }
    for (size_t i = 0; i < conn_size; ++i) {
      uintV u = source_conn[i];
      assert(vis[u]);
    }
    assert(vis[source_vertex]);
    //#endif

    Init(conn_vec, source_vertex, source_conn_vec, mapping, inverted_mapping);
  }

  void Init(ConnType& conn, uintV source_vertex, ConnType& source_conn,
            VertexMapping& mapping, VertexMapping& inverted_mapping) {
    conn_.assign(conn.begin(), conn.end());
    source_vertex_ = source_vertex;
    source_conn_.assign(source_conn.begin(), source_conn.end());
    mapping_.assign(mapping.begin(), mapping.end());
    inverted_mapping_.assign(inverted_mapping.begin(), inverted_mapping.end());

    assert(conn_.size() == source_conn_.size());
    std::sort(conn_.begin(), conn_.end());
    std::sort(source_conn_.begin(), source_conn_.end());
  }

  // transform from original vertex id to the index in the match order
  void TransformIndexBased(const SearchSequence& seq) {
    size_t n = seq.size();
    std::vector<uintV> otn(n, n);
    for (uintV u = 0; u < seq.size(); ++u) {
      otn[seq[u]] = u;
    }

    for (size_t i = 0; i < conn_.size(); ++i) {
      conn_[i] = otn[conn_[i]];
    }
    for (size_t i = 0; i < source_conn_.size(); ++i) {
      source_conn_[i] = otn[source_conn_[i]];
    }
    source_vertex_ = otn[source_vertex_];

    VertexMapping new_mapping(n, n);
    for (uintV u = 0; u < n; ++u) {
      if (mapping_[u] < n) {
        new_mapping[otn[u]] = otn[mapping_[u]];
      }
    }
    mapping_.swap(new_mapping);

    inverted_mapping_.clear();
    inverted_mapping_.resize(n, n);
    for (uintV u = 0; u < n; ++u) {
      if (mapping_[u] < n) {
        inverted_mapping_[mapping_[u]] = u;
      }
    }
  }

  void Print() const {
    std::cout << "[source_vertex=" << source_vertex_;
    std::cout << ",conn=(";
    for (size_t i = 0; i < conn_.size(); ++i) {
      if (i > 0) std::cout << ",";
      std::cout << conn_[i];
    }
    std::cout << "),source_conn=(";
    for (size_t i = 0; i < source_conn_.size(); ++i) {
      if (i > 0) std::cout << ",";
      std::cout << source_conn_[i];
    }
    std::cout << ")]";
  }

  ///////   getters function ////////
  ConnType& GetConnectivity() { return conn_; }
  uintV GetSourceVertex() const { return source_vertex_; }
  ConnType& GetSourceConnectivity() { return source_conn_; }
  VertexMapping& GetMapping() { return mapping_; }
  VertexMapping& GetInvertedMapping() { return inverted_mapping_; }
  uintV GetAlignedVertex() const {
    size_t aligned_vertex = source_vertex_;
    for (size_t i = 0; i < conn_.size(); ++i) {
      if (conn_[i] != source_conn_[i]) {
        aligned_vertex = source_conn_[i];
        break;
      }
    }
    return aligned_vertex;
  }

  const ConnType& GetConnectivity() const { return conn_; }
  const ConnType& GetSourceConnectivity() const { return source_conn_; }
  const VertexMapping& GetMapping() const { return mapping_; }
  const VertexMapping& GetInvertedMapping() const { return inverted_mapping_; }

  // a set of connectivity to be reused
  ConnType conn_;
  // the source pattern vertex where the reused result comes from
  uintV source_vertex_;
  // the connectivity that source_vertex_ has
  ConnType source_conn_;
  // one to one mapping : source_conn_ -> conn_
  VertexMapping mapping_;
  // the reverse of mapping_, note that some in conn_ may not have entry in
  // inverted_mapping_
  VertexMapping inverted_mapping_;
};

// For a pattern vertex, the plan to generate the instances
struct VertexReuseIntersectPlan {
  VertexReuseIntersectPlan() {}

  // transform from original vertex id to the index in the match order
  void TransformIndexBased(const SearchSequence& seq) {
    for (auto& reuse_conn_meta : reuse_conn_meta_) {
      reuse_conn_meta.TransformIndexBased(seq);
    }

    size_t n = seq.size();
    std::vector<uintV> otn(n, n);
    for (uintV u = 0; u < seq.size(); ++u) {
      otn[seq[u]] = u;
    }
    for (size_t i = 0; i < separate_conn_.size(); ++i) {
      separate_conn_[i] = otn[separate_conn_[i]];
    }
  }

  void Print() const {
    std::cout << "intersect: ";
    size_t count = 0;
    for (size_t i = 0; i < reuse_conn_meta_.size(); ++i) {
      if (count) std::cout << " ";
      reuse_conn_meta_[i].Print();
      ++count;
    }
    for (size_t i = 0; i < separate_conn_.size(); ++i) {
      if (count) std::cout << " ";
      std::cout << separate_conn_[i];
      ++count;
    }
  }

  // //// getter function ///////
  std::vector<ReuseConnMeta>& GetReuseConnectivityMeta() {
    return reuse_conn_meta_;
  }
  const std::vector<ReuseConnMeta>& GetReuseConnectivityMeta() const {
    return reuse_conn_meta_;
  }

  ConnType& GetSeparateConnectivity() { return separate_conn_; }
  const ConnType& GetSeparateConnectivity() const { return separate_conn_; }

  // Toe perform intersection, we reuse some intersection result from
  // reuse_conn_meta_, and some adjacent lists of some data vertics.
  // reuse_conn_meta_ indicates the reused intersection result
  // separate_conn_: each separate pattern vertex to intersect
  std::vector<ReuseConnMeta> reuse_conn_meta_;
  ConnType separate_conn_;
};

typedef std::vector<VertexReuseIntersectPlan> LevelReuseIntersectPlan;

#endif
