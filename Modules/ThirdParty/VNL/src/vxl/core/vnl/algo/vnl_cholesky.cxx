// This is core/vnl/algo/vnl_cholesky.cxx
//:
// \file
// vnl_cholesky
// \author Andrew W. Fitzgibbon, Oxford RRG
// Created: 08 Dec 96
//
//-----------------------------------------------------------------------------

#include <cmath>
#include <cassert>
#include <iostream>
#include "vnl_cholesky.h"
#include <vnl/algo/vnl_netlib.h> // dpofa_(), dposl_(), dpoco_(), dpodi_(), spofa_(), sposl_(), spoco_(), spodi_()

//: Cholesky decomposition.
// Make cholesky decomposition of M optionally computing
// the reciprocal condition number.  If mode is estimate_condition, the
// condition number and an approximate nullspace are estimated, at a cost
// of a factor of (1 + 18/n).  Here's a table of 1 + 18/n:
// \verbatim
// n:              3      5     10     50    100    500   1000
// slowdown:     7.0    4.6    2.8    1.4   1.18   1.04   1.02
// \endverbatim

template <typename T>
vnl_cholesky<T>::vnl_cholesky(vnl_matrix<T> const & M, Operation mode)
  : A_(M)
{
  long n = M.columns();
  assert(n == (int)(M.rows()));
  num_dims_rank_def_ = -1;
  if (std::fabs(M(0, n - 1) - M(n - 1, 0)) > 1e-8)
  {
    std::cerr << "vnl_cholesky: WARNING: non-symmetric: " << M << std::endl;
  }

  if (mode != estimate_condition)
  {
    // Quick factorization
    if constexpr (std::is_same<T, double>::value)
    {
      v3p_netlib_dpofa_(A_.data_block(), &n, &n, &num_dims_rank_def_);
    }
    else if constexpr (std::is_same<T, float>::value)
    {
      v3p_netlib_spofa_(A_.data_block(), &n, &n, &num_dims_rank_def_);
    }
    if (mode == verbose && num_dims_rank_def_ != 0)
      std::cerr << "vnl_cholesky: " << num_dims_rank_def_ << " dimensions of non-posdeffness\n";
  }
  else
  {
    vnl_vector<T> nullvec(n);
    if constexpr (std::is_same<T, double>::value)
    {
      v3p_netlib_dpoco_(A_.data_block(), &n, &n, &rcond_, nullvec.data_block(), &num_dims_rank_def_);
    }
    else if constexpr (std::is_same<T, float>::value)
    {
      v3p_netlib_spoco_(A_.data_block(), &n, &n, &rcond_, nullvec.data_block(), &num_dims_rank_def_);
    }
    if (num_dims_rank_def_ != 0)
      std::cerr << "vnl_cholesky: rcond=" << rcond_ << " so " << num_dims_rank_def_
                << " dimensions of non-posdeffness\n";
  }
}

//: Solve least squares problem M x = b.
//  The right-hand-side std::vector x may be b,
//  which will give a fractional increase in speed.
template <typename T>
void
vnl_cholesky<T>::solve(vnl_vector<T> const & b, vnl_vector<T> * x) const
{
  assert(b.size() == A_.columns());

  *x = b;
  long n = A_.columns();
  if constexpr (std::is_same<T, double>::value)
  {
    v3p_netlib_dposl_(A_.data_block(), &n, &n, x->data_block());
  }
  else if constexpr (std::is_same<T, float>::value)
  {
    v3p_netlib_sposl_(A_.data_block(), &n, &n, x->data_block());
  }
}

//: Solve least squares problem M x = b.
template <typename T>
vnl_vector<T>
vnl_cholesky<T>::solve(vnl_vector<T> const & b) const
{
  assert(b.size() == A_.columns());

  long n = A_.columns();
  vnl_vector<T> ret = b;
  if constexpr (std::is_same<T, double>::value)
  {
    v3p_netlib_dposl_(A_.data_block(), &n, &n, ret.data_block());
  }
  else if constexpr (std::is_same<T, float>::value)
  {
    v3p_netlib_sposl_(A_.data_block(), &n, &n, ret.data_block());
  }
  return ret;
}

//: Compute determinant.
template <typename T>
T vnl_cholesky<T>::determinant() const
{
  long n = A_.columns();
  vnl_matrix<T> I = A_;
  T det[2];
  long job = 10;
  if constexpr (std::is_same<T, double>::value)
  {
    v3p_netlib_dpodi_(I.data_block(), &n, &n, det, &job);
  }
  else if constexpr (std::is_same<T, float>::value)
  {
    v3p_netlib_spodi_(I.data_block(), &n, &n, det, &job);
  }
  return det[0] * std::pow(10.0, det[1]);
}

// : Compute inverse.  Not efficient.
template <typename T>
vnl_matrix<T>
vnl_cholesky<T>::inverse() const
{
  if (num_dims_rank_def_)
  {
    std::cerr << "vnl_cholesky: Calling inverse() on rank-deficient matrix\n";
    return vnl_matrix<T>();
  }

  long n = A_.columns();
  vnl_matrix<T> I = A_;
  long job = 01;
  if constexpr (std::is_same<T, double>::value)
  {
    v3p_netlib_dpodi_(I.data_block(), &n, &n, nullptr, &job);
  }
  else if constexpr (std::is_same<T, float>::value)
  {
    v3p_netlib_spodi_(I.data_block(), &n, &n, nullptr, &job);
  }

  // Copy lower triangle into upper
  for (int i = 0; i < n; ++i)
    for (int j = i + 1; j < n; ++j)
      I(i, j) = I(j, i);

  return I;
}

//: Return lower-triangular factor.
template <typename T>
vnl_matrix<T>
vnl_cholesky<T>::lower_triangle() const
{
  unsigned n = A_.columns();
  vnl_matrix<T> L(n, n);
  // Zap upper triangle and transpose
  for (unsigned i = 0; i < n; ++i)
  {
    L(i, i) = A_(i, i);
    for (unsigned j = i + 1; j < n; ++j)
    {
      L(j, i) = A_(j, i);
      L(i, j) = 0;
    }
  }
  return L;
}


//: Return upper-triangular factor.
template <typename T>
vnl_matrix<T>
vnl_cholesky<T>::upper_triangle() const
{
  unsigned n = A_.columns();
  vnl_matrix<T> U(n, n);
  // Zap lower triangle and transpose
  for (unsigned i = 0; i < n; ++i)
  {
    U(i, i) = A_(i, i);
    for (unsigned j = i + 1; j < n; ++j)
    {
      U(i, j) = A_(j, i);
      U(j, i) = 0;
    }
  }
  return U;
}

template class vnl_cholesky<float>;
template class vnl_cholesky<double>;
