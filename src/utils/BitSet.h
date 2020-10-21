#ifndef __UTILS_BITSET_H__
#define __UTILS_BITSET_H__

#include <vector>

class BitSet {
 public:
  BitSet(size_t count) { bitmaps_.resize(count, false); }
  BitSet(BitSet& obj) {
    size_t sz = obj.Size();
    bitmaps_.resize(sz);
    for (size_t i = 0; i < sz; ++i) {
      bitmaps_[i] = obj.Get(i);
    }
  }
  void Set(size_t idx, bool v) { bitmaps_[idx] = v; }
  void Reset(bool v) {
    for (size_t i = 0; i < bitmaps_.size(); ++i) {
      bitmaps_[i] = v;
    }
  }
  bool Get(size_t idx) const { return bitmaps_.at(idx); }
  size_t Count() const {
    size_t ret = 0;
    for (size_t i = 0; i < bitmaps_.size(); ++i) {
      if (bitmaps_[i]) ret++;
    }
    return ret;
  }
  bool All() const { return this->Count() == bitmaps_.size(); }
  bool Any() const { return this->Count() > 0; }
  size_t Size() const { return bitmaps_.size(); }
  bool Equal(BitSet& obj) const {
    if (bitmaps_.size() != obj.Size()) return false;
    for (size_t i = 0; i < bitmaps_.size(); ++i) {
      if (bitmaps_[i] != obj.Get(i)) {
        return false;
      }
    }
    return true;
  }

 protected:
  std::vector<bool> bitmaps_;
};

#endif
