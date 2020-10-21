#ifndef __QUERY_H__
#define __QUERY_H__

#include <cassert>
#include <sstream>
#include <string>
#include <vector>
#include "Meta.h"
#include "QueryCommon.h"

class Query {
 public:
  Query(QueryType query_type, bool enable_ordering = true)
      : query_type_(query_type), enable_ordering_(enable_ordering) {
    if (query_type_ == Q0) {
      Q0Query();
    } else if (query_type_ == Q1) {
      Q1Query();
    } else if (query_type_ == Q2) {
      Q2Query();
    } else if (query_type_ == Q3) {
      Q3Query();
    } else if (query_type_ == Q4) {
      Q4Query();
    } else if (query_type_ == Q5) {
      Q5Query();
    } else if (query_type_ == Q6) {
      Q6Query();
    } else if (query_type_ == Q7) {
      Q7Query();
    } else if (query_type_ == Q8) {
      Q8Query();
    } else if (query_type_ == Q9) {
      Q9Query();
    } else if (query_type_ == Q10) {
      Q10Query();
    } else if (query_type_ == Q11) {
      Q11Query();
    } else if (query_type_ == Q12) {
      Q12Query();
    } else if (query_type_ == Q13) {
      Q13Query();
    } else if (query_type_ == LINE) {
      // for preprocessing edge weight
      // do nothing here
      vertex_count_ = 0;
    } else {
      assert(false);
    }
    if (!enable_ordering_) {
      DisableOrdering();
    }
  }

  std::string GetString() const {
    std::string s = "Q";
    std::ostringstream temp;
    temp << query_type_;
    s += temp.str();
    return s;
  }

  QueryType GetQueryType() const { return query_type_; }
  size_t GetVertexCount() const { return vertex_count_; }
  AllConnType& GetConnectivity() { return con_; }
  AllCondType& GetOrdering() { return order_; }
  bool GetEnableOrdering() const { return enable_ordering_; }

 private:
  QueryType query_type_;
  size_t vertex_count_;
  AllConnType con_;
  AllCondType order_;
  bool enable_ordering_;

 private:
  void DisableOrdering() {
    order_.resize(vertex_count_);
    for (size_t i = 0; i < vertex_count_; ++i) {
      order_[i].clear();
    }
  }

  void Q0Query() {
    vertex_count_ = 3;

    con_.resize(vertex_count_);
    con_[0].push_back(1);
    con_[0].push_back(2);
    con_[1].push_back(0);
    con_[1].push_back(2);
    con_[2].push_back(0);
    con_[2].push_back(1);

    order_.resize(vertex_count_);
    order_[0].push_back(std::make_pair(LESS_THAN, 1));
    order_[1].push_back(std::make_pair(LARGER_THAN, 0));
    order_[1].push_back(std::make_pair(LESS_THAN, 2));
    order_[2].push_back(std::make_pair(LARGER_THAN, 1));
  }
  void Q1Query() {
    vertex_count_ = 4;

    con_.resize(vertex_count_);
    con_[0].push_back(1);
    con_[0].push_back(3);
    con_[1].push_back(0);
    con_[1].push_back(2);
    con_[2].push_back(1);
    con_[2].push_back(3);
    con_[3].push_back(0);
    con_[3].push_back(2);

    order_.resize(vertex_count_);
    order_[0].push_back(std::make_pair(LESS_THAN, 1));
    order_[0].push_back(std::make_pair(LESS_THAN, 2));
    order_[0].push_back(std::make_pair(LESS_THAN, 3));
    order_[1].push_back(std::make_pair(LARGER_THAN, 0));
    order_[1].push_back(std::make_pair(LESS_THAN, 3));
    order_[2].push_back(std::make_pair(LARGER_THAN, 0));
    order_[3].push_back(std::make_pair(LARGER_THAN, 0));
    order_[3].push_back(std::make_pair(LARGER_THAN, 1));
  }
  void Q2Query() {
    vertex_count_ = 4;

    con_.resize(vertex_count_);
    con_[0].push_back(1);
    con_[0].push_back(2);
    con_[0].push_back(3);
    con_[1].push_back(0);
    con_[1].push_back(2);
    con_[2].push_back(0);
    con_[2].push_back(1);
    con_[2].push_back(3);
    con_[3].push_back(0);
    con_[3].push_back(2);

    order_.resize(vertex_count_);
    order_[0].push_back(std::make_pair(LESS_THAN, 2));
    order_[1].push_back(std::make_pair(LESS_THAN, 3));
    order_[2].push_back(std::make_pair(LARGER_THAN, 0));
    order_[3].push_back(std::make_pair(LARGER_THAN, 1));
  }
  void Q3Query() {
    vertex_count_ = 4;

    con_.resize(vertex_count_);
    con_[0].push_back(1);
    con_[0].push_back(2);
    con_[0].push_back(3);
    con_[1].push_back(0);
    con_[1].push_back(2);
    con_[1].push_back(3);
    con_[2].push_back(0);
    con_[2].push_back(1);
    con_[2].push_back(3);
    con_[3].push_back(0);
    con_[3].push_back(1);
    con_[3].push_back(2);

    order_.resize(vertex_count_);
    order_[0].push_back(std::make_pair(LESS_THAN, 1));
    order_[1].push_back(std::make_pair(LARGER_THAN, 0));
    order_[1].push_back(std::make_pair(LESS_THAN, 2));
    order_[2].push_back(std::make_pair(LARGER_THAN, 1));
    order_[2].push_back(std::make_pair(LESS_THAN, 3));
    order_[3].push_back(std::make_pair(LARGER_THAN, 2));
  }
  void Q4Query() {
    vertex_count_ = 5;

    con_.resize(5);
    con_[0].push_back(1);
    con_[0].push_back(2);
    con_[1].push_back(0);
    con_[1].push_back(2);
    con_[1].push_back(3);
    con_[2].push_back(0);
    con_[2].push_back(1);
    con_[2].push_back(4);
    con_[3].push_back(1);
    con_[3].push_back(4);
    con_[4].push_back(2);
    con_[4].push_back(3);

    order_.resize(vertex_count_);
    order_[1].push_back(std::make_pair(LESS_THAN, 2));
    order_[2].push_back(std::make_pair(LARGER_THAN, 1));
  }

  void Q5Query() {
    vertex_count_ = 6;

    con_.resize(vertex_count_);
    con_[0].push_back(1);
    con_[0].push_back(2);
    con_[0].push_back(3);
    con_[0].push_back(4);
    con_[0].push_back(5);
    con_[1].push_back(0);
    con_[1].push_back(2);
    con_[2].push_back(0);
    con_[2].push_back(1);
    con_[2].push_back(3);
    con_[3].push_back(0);
    con_[3].push_back(2);
    con_[3].push_back(4);
    con_[4].push_back(0);
    con_[4].push_back(3);
    con_[4].push_back(5);
    con_[5].push_back(0);
    con_[5].push_back(4);

    order_.resize(vertex_count_);
    order_[2].push_back(std::make_pair(LESS_THAN, 4));
    order_[4].push_back(std::make_pair(LARGER_THAN, 2));
  }

  void Q6Query() {
    vertex_count_ = 5;

    con_.resize(vertex_count_);
    con_[0].push_back(1);
    con_[0].push_back(2);
    con_[0].push_back(3);
    con_[0].push_back(4);
    con_[1].push_back(0);
    con_[1].push_back(2);
    con_[1].push_back(3);
    con_[2].push_back(0);
    con_[2].push_back(1);
    con_[2].push_back(3);
    con_[3].push_back(0);
    con_[3].push_back(1);
    con_[3].push_back(2);
    con_[3].push_back(4);
    con_[4].push_back(0);
    con_[4].push_back(3);

    order_.resize(vertex_count_);
    order_[0].push_back(std::make_pair(LESS_THAN, 3));
    order_[3].push_back(std::make_pair(LARGER_THAN, 0));
    order_[1].push_back(std::make_pair(LESS_THAN, 2));
    order_[2].push_back(std::make_pair(LARGER_THAN, 1));
  }

  void Q7Query() {
    vertex_count_ = 5;

    con_.resize(vertex_count_);
    con_[0].push_back(1);
    con_[0].push_back(2);
    con_[0].push_back(3);
    con_[0].push_back(4);
    con_[1].push_back(0);
    con_[1].push_back(2);
    con_[1].push_back(3);
    con_[1].push_back(4);
    con_[2].push_back(0);
    con_[2].push_back(1);
    con_[2].push_back(3);
    con_[2].push_back(4);
    con_[3].push_back(0);
    con_[3].push_back(1);
    con_[3].push_back(2);
    con_[3].push_back(4);
    con_[4].push_back(0);
    con_[4].push_back(1);
    con_[4].push_back(2);
    con_[4].push_back(3);

    order_.resize(vertex_count_);
    order_[0].push_back(std::make_pair(LESS_THAN, 1));
    order_[1].push_back(std::make_pair(LARGER_THAN, 0));
    order_[1].push_back(std::make_pair(LESS_THAN, 2));
    order_[2].push_back(std::make_pair(LARGER_THAN, 1));
    order_[2].push_back(std::make_pair(LESS_THAN, 3));
    order_[3].push_back(std::make_pair(LARGER_THAN, 2));
    order_[3].push_back(std::make_pair(LESS_THAN, 4));
    order_[4].push_back(std::make_pair(LARGER_THAN, 3));
  }
  void Q8Query() {
    vertex_count_ = 5;

    con_.resize(vertex_count_);
    con_[0].push_back(1);
    con_[0].push_back(2);
    con_[0].push_back(3);
    con_[1].push_back(0);
    con_[1].push_back(2);
    con_[1].push_back(3);
    con_[2].push_back(0);
    con_[2].push_back(1);
    con_[2].push_back(4);
    con_[3].push_back(0);
    con_[3].push_back(1);
    con_[3].push_back(4);
    con_[4].push_back(2);
    con_[4].push_back(3);

    order_.resize(vertex_count_);
    order_[0].push_back(std::make_pair(LESS_THAN, 1));
    order_[1].push_back(std::make_pair(LARGER_THAN, 0));
    order_[2].push_back(std::make_pair(LESS_THAN, 3));
    order_[3].push_back(std::make_pair(LARGER_THAN, 2));
  }
  void Q9Query() {
    vertex_count_ = 5;

    con_.resize(vertex_count_);
    for (uintV i = 1; i <= 4; ++i) {
      con_[0].push_back(i);
      con_[i].push_back(0);
    }
    con_[1].push_back(3);
    con_[3].push_back(1);
    con_[1].push_back(2);
    con_[2].push_back(1);
    con_[2].push_back(4);
    con_[4].push_back(2);
    order_.resize(vertex_count_);
    order_[1].push_back(std::make_pair(LESS_THAN, 2));
    order_[2].push_back(std::make_pair(LARGER_THAN, 1));
  }
  void Q10Query() {
    vertex_count_ = 5;

    con_.resize(vertex_count_);
    con_[0].push_back(1);
    con_[0].push_back(2);
    con_[0].push_back(3);
    con_[0].push_back(4);
    con_[1].push_back(0);
    con_[1].push_back(2);
    con_[1].push_back(4);
    con_[2].push_back(0);
    con_[2].push_back(1);
    con_[2].push_back(3);
    con_[3].push_back(0);
    con_[3].push_back(2);
    con_[3].push_back(4);
    con_[4].push_back(0);
    con_[4].push_back(1);
    con_[4].push_back(3);

    order_.resize(vertex_count_);
    order_[1].push_back(std::make_pair(LESS_THAN, 2));
    order_[1].push_back(std::make_pair(LESS_THAN, 3));
    order_[1].push_back(std::make_pair(LESS_THAN, 4));
    order_[2].push_back(std::make_pair(LESS_THAN, 4));
    order_[2].push_back(std::make_pair(LARGER_THAN, 1));
    order_[3].push_back(std::make_pair(LARGER_THAN, 1));
    order_[4].push_back(std::make_pair(LARGER_THAN, 1));
    order_[4].push_back(std::make_pair(LARGER_THAN, 2));
  }

  void Q11Query() {
    vertex_count_ = 6;
    con_.resize(vertex_count_);
    for (uintV i = 0; i < vertex_count_; ++i) {
      for (uintV j = 0; j < vertex_count_; ++j) {
        if (j != i) {
          con_[i].push_back(j);
        }
      }
    }
    order_.resize(vertex_count_);
    for (uintV i = 0; i + 1 < vertex_count_; ++i) {
      order_[i].push_back(std::make_pair(LESS_THAN, i + 1));
      order_[i + 1].push_back(std::make_pair(LARGER_THAN, i));
    }
  }

  void Q12Query() {
    vertex_count_ = 5;

    con_.resize(vertex_count_);
    con_[0].push_back(1);
    con_[0].push_back(2);
    con_[0].push_back(3);
    con_[0].push_back(4);
    con_[1].push_back(0);
    con_[1].push_back(2);
    con_[1].push_back(3);
    con_[1].push_back(4);
    con_[3].push_back(0);
    con_[3].push_back(1);
    con_[3].push_back(2);
    con_[3].push_back(4);
    con_[2].push_back(0);
    con_[2].push_back(1);
    con_[2].push_back(3);
    con_[4].push_back(0);
    con_[4].push_back(1);
    con_[4].push_back(3);

    order_.resize(vertex_count_);
    order_[0].push_back(std::make_pair(LESS_THAN, 1));
    order_[2].push_back(std::make_pair(LESS_THAN, 4));
    order_[1].push_back(std::make_pair(LARGER_THAN, 0));
    order_[4].push_back(std::make_pair(LARGER_THAN, 2));
  }

  void Q13Query() {
    vertex_count_ = 6;

    con_.resize(vertex_count_);
    con_[0].push_back(1);
    con_[0].push_back(2);
    con_[0].push_back(3);
    con_[0].push_back(5);
    con_[1].push_back(0);
    con_[1].push_back(2);
    con_[1].push_back(3);
    con_[1].push_back(4);
    con_[2].push_back(0);
    con_[2].push_back(1);
    con_[2].push_back(4);
    con_[2].push_back(5);
    con_[3].push_back(0);
    con_[3].push_back(1);
    con_[4].push_back(1);
    con_[4].push_back(2);
    con_[5].push_back(0);
    con_[5].push_back(2);

    order_.resize(vertex_count_);
    order_[0].push_back(std::make_pair(LESS_THAN, 1));
    order_[1].push_back(std::make_pair(LESS_THAN, 2));
    order_[1].push_back(std::make_pair(LARGER_THAN, 0));
    order_[2].push_back(std::make_pair(LARGER_THAN, 1));
  }
};

#endif
