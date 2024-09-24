// This is core/vnl/algo/vnl_cholesky.h
#ifndef vnl_cholesky_h_
#define vnl_cholesky_h_
//:
// \file
// \brief Decomposition of symmetric matrix
// \author Andrew W. Fitzgibbon, Oxford RRG
// \date   08 Dec 96
//
// \verbatim
//  Modifications
//   Peter Vanroose, Leuven, Apr 1998: added L() (return decomposition matrix)
//   dac (Manchester) 26/03/2001: tidied up documentation
//   Feb.2002 - Peter Vanroose - brief doxygen comment placed on single line
// \endverbatim

#include <vnl/vnl_vector.h>
#include <vnl/vnl_matrix.h>
#include <vnl/algo/vnl_algo_export.h>

//: Decomposition of symmetric matrix.
//  A class to hold the Cholesky decomposition of a symmetric matrix and
//  use that to solve linear systems, compute determinants and inverses.
//  The cholesky decomposition decomposes symmetric A = L*L.transpose()
//  where L is lower triangular
//
//  To check that the decomposition can be used safely for solving a linear
//  equation it is wise to construct with mode==estimate_condition and
//  check that rcond()>sqrt(machine precision).  If this is not the case
//  it might be a good idea to use vnl_svd instead.
template <typename T = double>
class VNL_ALGO_EXPORT vnl_cholesky
{
 public:
  //: Modes of computation.  See constructor for details.
  enum Operation {
    quiet,
    verbose,
    estimate_condition
  };

  //: Make cholesky decomposition of M optionally computing the reciprocal condition number.
  vnl_cholesky(vnl_matrix<T> const& M, Operation mode = verbose);
 ~vnl_cholesky() = default;

  //: Solve LS problem M x = b
  vnl_vector<T> solve(vnl_vector<T> const& b) const;

  //: Solve LS problem M x = b
  void solve(vnl_vector<T> const& b, vnl_vector<T>* x) const;

  //: Compute determinant
  T determinant() const;

  //   Compute inverse.  Not efficient.
  // It's broken, I don't have time to fix it.
  // Mail awf@robots if you need it and I'll tell you as much as I can
  // to fix it.
  vnl_matrix<T> inverse() const;

  //: Return lower-triangular factor.
  vnl_matrix<T> lower_triangle() const;

  //: Return upper-triangular factor.
  vnl_matrix<T> upper_triangle() const;

  //: Return the decomposition matrix
  vnl_matrix<T> const& L_badly_named_method() const { return A_; }

  //: A Success/failure flag
  int rank_deficiency() const { return num_dims_rank_def_; }

  //: Return reciprocal condition number (smallest/largest singular values).
  // As long as rcond()>sqrt(precision) the decomposition can be used for
  // solving equations safely.
  // Not calculated unless Operation mode at construction was estimate_condition.
  T rcond() const { return rcond_; }

  //: Return computed nullvector.
  // Not calculated unless Operation mode at construction was estimate_condition.
  vnl_vector<T>      & nullvector()       { return nullvector_; }
  vnl_vector<T> const& nullvector() const { return nullvector_; }

 protected:
  // Data Members--------------------------------------------------------------
  vnl_matrix<T> A_;
  T rcond_;
  long num_dims_rank_def_;
  vnl_vector<T> nullvector_;

 private:
  //: Copy constructor - privatised to avoid it being used
  vnl_cholesky(vnl_cholesky const & that) = delete;
  //: Assignment operator - privatised to avoid it being used
  vnl_cholesky& operator=(vnl_cholesky const & that) = delete;
};

#endif // vnl_cholesky_h_
