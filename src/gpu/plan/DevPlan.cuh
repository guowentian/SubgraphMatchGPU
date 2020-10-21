#ifndef __DEV_PLAN_CUH__
#define __DEV_PLAN_CUH__

#include <iostream>
#include "CudaContext.cuh"
#include "GPUUtil.cuh"
#include "Plan.h"

template <int SIZE, typename T>
struct DevArrayType {
  // Restrict DevArrayType to be initiailized from the vector in the host side
  HOST void Set(std::vector<T>& from) {
    count_ = from.size();
    for (size_t i = 0; i < from.size(); ++i) {
      array_[i] = from[i];
    }
  }

  friend std::ostream& operator<<(std::ostream& out,
                                  const DevArrayType<SIZE, T>& from) {
    out << "[count=" << from.GetCount() << ",data=(";
    for (size_t j = 0; j < from.GetCount(); ++j) {
      if (j > 0) out << ",";
      out << from.Get(j);
    }
    out << ")]" << std::endl;
    return out;
  }

  HOST void Print() const { std::cout << *this; }

  // getter
  HOST_DEVICE size_t GetCount() const { return count_; }
  HOST_DEVICE T* GetArray() const { return array_; }
  HOST_DEVICE T Get(size_t index) const { return array_[index]; }
  // Return by reference is needed because T may be an array
  HOST_DEVICE T& Get(size_t index) { return array_[index]; }

  T array_[SIZE];
  size_t count_;
};

// Adjacent list of one vertex
struct DevConnType : DevArrayType<kMaxQueryVerticesNum, uintV> {
  // To support the legacy code that call GetConnectivity().
  // This API is actually the same as Get(int index)
  HOST_DEVICE uintV GetConnectivity(size_t index) const {
    return array_[index];
  }
};
// Adjacent lists of all vertices
struct DevAllConnType : DevArrayType<kMaxQueryVerticesNum, DevConnType> {
  HOST void Set(std::vector<std::vector<uintV>>& from) {
    count_ = from.size();
    for (size_t i = 0; i < from.size(); ++i) {
      array_[i].Set(from[i]);
    }
  }
};

struct DevCondType {
  // DevCondType can be the type T of DevArrayType, so operator= and << are
  // required
  HOST DevCondType& operator=(const DevCondType& from) {
    operator_ = from.GetOperator();
    operand_ = from.GetOperand();
    return *this;
  }
  // support initialization from CondType in the host side
  HOST DevCondType& operator=(const CondType& from) {
    operator_ = from.first;
    operand_ = from.second;
    return *this;
  }

  friend std::ostream& operator<<(std::ostream& out, const DevCondType& from) {
    std::string op_str;
    switch (from.GetOperator()) {
      case LESS_THAN:
        op_str = "LESS_THAN";
        break;
      case LARGER_THAN:
        op_str = "LARGER_THAN";
        break;
      case NON_EQUAL:
        op_str = "NON_EQUAL";
        break;
      default:
        break;
    }
    out << "<" << op_str << "," << from.GetOperand() << ">";
    return out;
  }

  HOST_DEVICE CondOperator GetOperator() const { return operator_; }
  HOST_DEVICE uintV GetOperand() const { return operand_; }

  CondOperator operator_;
  uintV operand_;
};

// Ordering constraint for one vertex
struct DevCondArrayType : DevArrayType<kMaxQueryVerticesNum, DevCondType> {
  HOST void Set(std::vector<CondType>& from) {
    count_ = from.size();
    for (size_t i = 0; i < from.size(); ++i) {
      array_[i] = from[i];
    }
  }

  // Support the legacy code that call GetCondition
  // This API is actually the same as Get(int index)
  HOST_DEVICE DevCondType GetCondition(size_t index) const {
    return array_[index];
  }
};

// Ordering constraint for all vertices
struct DevAllCondType : DevArrayType<kMaxQueryVerticesNum, DevCondArrayType> {
  HOST void Set(std::vector<std::vector<CondType>>& from) {
    count_ = from.size();
    for (size_t i = 0; i < from.size(); ++i) {
      array_[i].Set(from[i]);
    }
  }
};

// A set of vertices stored in array
typedef DevArrayType<kMaxQueryVerticesNum, uintV> DevVTGroup;

#endif
