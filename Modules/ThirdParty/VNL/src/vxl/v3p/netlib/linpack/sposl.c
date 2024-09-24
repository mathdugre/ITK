/* sposl.f -- translated by f2c (version 20240504).
   You must link the resulting object file with libf2c:
  on Microsoft Windows system, link with libf2c.lib;
  on Linux or Unix systems, link with .../path/to/libf2c.a -lm
  or, if you install libf2c.a in a standard place, with -lf2c -lm
  -- in that order, at the end of the command line, as in
    cc *.o -lf2c -lm
  Source for libf2c is in /netlib/f2c/libf2c.zip, e.g.,

    http://www.netlib.org/f2c/libf2c.zip
*/

#ifdef __cplusplus
extern "C"
{
#endif
#include "v3p_netlib.h"

  /* Table of constant values */

  static integer c__1 = 1;

  /* Subroutine */ int
  sposl_(real * a, integer * lda, integer * n, real * b)
  {
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2;

    /* Local variables */
    static integer              k;
    static real                 t;
    static integer              kb;
    extern doublereal           sdot_(integer *, real *, integer *, real *, integer *);
    extern /* Subroutine */ int saxpy_(integer *, real *, real *, integer *, real *, integer *);


    /*     sposl solves the real symmetric positive definite system */
    /*     a * x = b */
    /*     using the factors computed by spoco or spofa. */

    /*     on entry */

    /*        a       real(lda, n) */
    /*                the output from spoco or spofa. */

    /*        lda     integer */
    /*                the leading dimension of the array  a . */

    /*        n       integer */
    /*                the order of the matrix  a . */

    /*        b       real(n) */
    /*                the right hand side vector. */

    /*     on return */

    /*        b       the solution vector  x . */

    /*     error condition */

    /*        a division by zero will occur if the input factor contains */
    /*        a zero on the diagonal.  technically this indicates */
    /*        singularity but it is usually caused by improper subroutine */
    /*        arguments.  it will not occur if the subroutines are called */
    /*        correctly and  info .eq. 0 . */

    /*     to compute  inverse(a) * c  where  c  is a matrix */
    /*     with  p  columns */
    /*           call spoco(a,lda,n,rcond,z,info) */
    /*           if (rcond is too small .or. info .ne. 0) go to ... */
    /*           do 10 j = 1, p */
    /*              call sposl(a,lda,n,c(1,j)) */
    /*        10 continue */

    /*     linpack.  this version dated 08/14/78 . */
    /*     cleve moler, university of new mexico, argonne national lab. */

    /*     subroutines and functions */

    /*     blas saxpy,sdot */

    /*     internal variables */


    /*     solve trans(r)*y = b */

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --b;

    /* Function Body */
    i__1 = *n;
    for (k = 1; k <= i__1; ++k)
    {
      i__2 = k - 1;
      t = sdot_(&i__2, &a[k * a_dim1 + 1], &c__1, &b[1], &c__1);
      b[k] = (b[k] - t) / a[k + k * a_dim1];
      /* L10: */
    }

    /*     solve r*x = y */

    i__1 = *n;
    for (kb = 1; kb <= i__1; ++kb)
    {
      k = *n + 1 - kb;
      b[k] /= a[k + k * a_dim1];
      t = -b[k];
      i__2 = k - 1;
      saxpy_(&i__2, &t, &a[k * a_dim1 + 1], &c__1, &b[1], &c__1);
      /* L20: */
    }
    return 0;
  } /* sposl_ */

#ifdef __cplusplus
}
#endif