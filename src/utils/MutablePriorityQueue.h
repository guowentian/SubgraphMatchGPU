#ifndef __UTILS_MUTABLE_PRIORITY_QUEUE_H__
#define __UTILS_MUTABLE_PRIORITY_QUEUE_H__

#include <cassert>
#include <iostream>
#include <type_traits>
#include <unordered_map>

template <typename KeyT, typename PriorityT>
struct PNode {
  PNode(KeyT k, PriorityT p) {
    key = k;
    pri = p;
  }
  PNode() {}
  KeyT key;
  PriorityT pri;
};

// an element has larger priority comes first
// the priority could be changed
// This priority queue requires the keys:
// #keys are required to be <= n
// the values of keys are in the range <=n
template <typename KeyType, typename TPriority>
class MutablePriorityQueue {
 public:
  MutablePriorityQueue(size_t n) {
    // <key, TPriority> pair
    static_assert(std::is_integral<KeyType>::value,
                  "Key type is required to be integer type");

    size = n;
    len = 0;
    priority = new TPriority[size + 1];
    index_to_key = new KeyType[size + 1];
    key_to_index = new size_t[size + 1];
    for (size_t i = 0; i <= size; ++i) {
      // indicate non-existence
      key_to_index[i] = size + 1;
    }
  }

  ~MutablePriorityQueue() {
    delete[] priority;
    priority = NULL;
    delete[] index_to_key;
    index_to_key = NULL;
    delete[] key_to_index;
    key_to_index = NULL;
  }

  size_t GetQueueSize() { return len; }

  void Insert(KeyType k, TPriority pri) {
    key_to_index[k] = ++len;
    index_to_key[len] = k;
    priority[len] = pri;
    MoveUp(len);
  }

  void ResetPriority(TPriority p) {
    for (size_t i = 0; i < size + 1; ++i) {
      priority[i] = p;
    }
  }

  void Clear() {
    len = 0;
    for (size_t i = 0; i <= size; ++i) {
      key_to_index[i] = size + 1;
    }
  }

  // Change the priority of a key,
  // if the key does not exist, insert it
  void ChangePriority(KeyType key, TPriority pri) {
    size_t idx = key_to_index[key];
    if (idx > size) {
      // if not exist, insert new one
      Insert(key, pri);
      return;
    }
    TPriority old_pri = priority[idx];
    priority[idx] = pri;
    if (CompareElement(pri, old_pri)) {
      // new priority is higher
      MoveUp(idx);
    } else {
      MoveDown(idx);
    }
  }

  TPriority GetPriority(KeyType key) {
    size_t idx = key_to_index[key];
    assert(idx <= size);
    return priority[idx];
  }

  // Remove the front element from the queue and return it
  PNode<KeyType, TPriority> Pop() {
    assert(len > 0);
    PNode<KeyType, TPriority> ret(index_to_key[1], priority[1]);
    SwapElement(1, len);
    len--;
    MoveDown(1);
    // indicate it does not exist any more
    key_to_index[ret.key] = size + 1;
    return ret;
  }

  void Remove(KeyType key) {
    size_t idx = key_to_index[key];
    // this key should exist
    assert(idx <= size);
    TPriority orig_pri = priority[idx];
    SwapElement(idx, len);
    len--;
    if (len > 0) {
      if (CompareElement(priority[idx], orig_pri)) {
        MoveUp(idx);
      } else {
        MoveDown(idx);
      }
      key_to_index[key] = size + 1;
    }
  }

  PNode<KeyType, TPriority> Peek() {
    // assert(len > 0);
    PNode<KeyType, TPriority> ret(index_to_key[1], priority[1]);
    return ret;
  }

  bool Exists(KeyType key) { return key_to_index[key] <= size; }

  // for debugging
  void Check() {
    for (size_t i = 1; i <= len; ++i) {
      size_t lc = LeftChild(i);
      if (lc < len) {
        assert(CompareElement(priority[i], priority[lc]));
      }
      size_t rc = RightChild(i);
      if (rc < len) {
        assert(CompareElement(priority[i], priority[rc]));
      }
    }
  }

 private:
  void MoveUp(size_t idx) {
    while (idx > 1) {
      size_t pidx = Parent(idx);
      if (CompareElement(priority[idx], priority[pidx])) {
        // idx has higher priority
        SwapElement(idx, pidx);
        idx = pidx;
      } else {
        break;
      }
    }
  }

  void MoveDown(size_t idx) {
    size_t lc = LeftChild(idx);
    size_t rc = RightChild(idx);
    while (lc <= len) {
      size_t max_child = lc;
      if (rc <= len && CompareElement(priority[rc], priority[lc])) {
        max_child = rc;
      }
      if (CompareElement(priority[max_child], priority[idx])) {
        // if max_child has higher priorty, swap, after swap, "max_child" has
        // higher priority than both of the children
        SwapElement(max_child, idx);
        idx = max_child;
      } else {
        break;
      }
      lc = LeftChild(idx);
      rc = RightChild(idx);
    }
  }

  // default comparator, the larger priority comes first
  inline bool CompareElement(TPriority p1, TPriority p2) { return p1 > p2; }

  void SwapElement(size_t i, size_t j) {
    // i, j are index
    size_t ki = index_to_key[i];
    size_t kj = index_to_key[j];
    std::swap(key_to_index[ki], key_to_index[kj]);
    std::swap(index_to_key[i], index_to_key[j]);
    std::swap(priority[i], priority[j]);
  }

  inline size_t LeftChild(size_t i) { return 2 * i; }
  inline size_t RightChild(size_t i) { return 2 * i + 1; }
  inline size_t Parent(size_t i) { return i / 2; }

 private:
  TPriority *priority;
  KeyType *index_to_key;
  size_t *key_to_index;
  size_t len;
  size_t size;
};

#endif
