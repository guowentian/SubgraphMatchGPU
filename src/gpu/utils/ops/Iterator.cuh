#ifndef __GPU_UTILS_OPERATIONS_ITERATORS_CUH__
#define __GPU_UTILS_OPERATIONS_ITERATORS_CUH__

#include <moderngpu/operators.hxx>

namespace GpuUtils {
namespace Iterator {

// CountingIterator is normally used to return the index directly.
// T is the value type returned by CountIterator.
// The type of the internal offset has to be set to int, because
// at line 200 of operators.hxx, the outer_t of const_iterator_t is
// set with int by default.

// Constructor: CountingIterator(T base).
// Example:
// CountingIterator iter(base); // base is of type T
// T v = iter[index] // v is equal to base+index
template <typename T>
using CountingIterator = mgpu::counting_iterator_t<T, int>;

// ConstantIterator always returns the initial value.
// T is the value type returned by ConstantIterator.
// The type of the internal offset is fixed to int, because at line 235 of
// operators.hxx, int is hardcoded.

// Constructor: ConstantIterator(T value)
// Example:
// ConstantIterator iter(base); // base is of type T
// T v = iter[index]; // v is always equal to base
template <typename T>
using ConstantIterator = mgpu::constant_iterator_t<T>;

// LoadIterator accepts an index and returns the output value of the functor
// ValueType is the return type of LoadFunctor
// We force the type of the internal offset to be size_t here.
// Example:
// LoadIterator<ValueType> iter = MakeLoadIterator(func, base); // ValueType is
// decided by the returned type of func; base is of type size_t
// ValueType v = iter[base + 10] // v is equal to func(base+10)
template <typename ValueType, typename LoadFunctor>
using LoadIterator =
    mgpu::lambda_iterator_t<LoadFunctor, mgpu::empty_t, ValueType, size_t>;

template <typename ValueType, typename LoadFunctor>
LoadIterator<ValueType, LoadFunctor> MakeLoadIterator(LoadFunctor load_functor,
                                                      size_t base) {
  return mgpu::make_load_iterator<ValueType, size_t>(load_functor, base);
}

}  // namespace Iterator
}  // namespace GpuUtils

#endif