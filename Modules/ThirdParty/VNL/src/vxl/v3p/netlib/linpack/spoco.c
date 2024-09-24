/* spoco.f -- translated by f2c (version 20240504).
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
  spoco_(real * a, integer * lda, integer * n, real * rcond, real * z__, integer * info)
  {
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2;
    real r__1, r__2;

    /* Builtin functions */
    double r_sign(real *, real *);

    /* Local variables */
    static integer i__, j, k;
    static real s, t;
    static integer kb;
    static real ek, sm, wk;
    static integer jm1, kp1;
    static real wkm;
    extern doublereal sdot_(integer *, real *, integer *, real *, integer *);
    extern /* Subroutine */ int sscal_(integer *, real *, real *, integer *), 
	    spofa_(real *, integer *, integer *, integer *);
    static real anorm;
    extern doublereal sasum_(integer *, real *, integer *);
    static real ynorm;
    extern /* Subroutine */ int saxpy_(integer *, real *, real *, integer *, 
	    real *, integer *);


/*     spoco factors a real symmetric positive definite matrix */
/*     and estimates the condition of the matrix. */

/*     if  rcond  is not needed, spofa is slightly faster. */
/*     to solve  a*x = b , follow spoco by sposl. */
/*     to compute  inverse(a)*c , follow spoco by sposl. */
/*     to compute  determinant(a) , follow spoco by spodi. */
/*     to compute  inverse(a) , follow spoco by spodi. */

/*     on entry */

/*        a       real(lda, n) */
/*                the symmetric matrix to be factored.  only the */
/*                diagonal and upper triangle are used. */

/*        lda     integer */
/*                the leading dimension of the array  a . */

/*        n       integer */
/*                the order of the matrix  a . */

/*     on return */

/*        a       an upper triangular matrix  r  so that  a = trans(r)*r */
/*                where  trans(r)  is the transpose. */
/*                the strict lower triangle is unaltered. */
/*                if  info .ne. 0 , the factorization is not complete. */

/*        rcond   real */
/*                an estimate of the reciprocal condition of  a . */
/*                for the system  a*x = b , relative perturbations */
/*                in  a  and  b  of size  epsilon  may cause */
/*                relative perturbations in  x  of size  epsilon/rcond . */
/*                if  rcond  is so small that the logical expression */
/*                           1.0 + rcond .eq. 1.0 */
/*                is true, then  a  may be singular to working */
/*                precision.  in particular,  rcond  is zero  if */
/*                exact singularity is detected or the estimate */
/*                underflows.  if info .ne. 0 , rcond is unchanged. */

/*        z       real(n) */
/*                a work vector whose contents are usually unimportant. */
/*                if  a  is close to a singular matrix, then  z  is */
/*                an approximate null vector in the sense that */
/*                norm(a*z) = rcond*norm(a)*norm(z) . */
/*                if  info .ne. 0 , z  is unchanged. */

/*        info    integer */
/*                = 0  for normal return. */
/*                = k  signals an error condition.  the leading minor */
/*                     of order  k  is not positive definite. */

/*     linpack.  this version dated 08/14/78 . */
/*     cleve moler, university of new mexico, argonne national lab. */

/*     subroutines and functions */

/*     linpack spofa */
/*     blas saxpy,sdot,sscal,sasum */
/*     fortran abs,amax1,real,sign */

/*     internal variables */



/*     find norm of a using only upper half */

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --z__;

    /* Function Body */
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	z__[j] = sasum_(&j, &a[j * a_dim1 + 1], &c__1);
	jm1 = j - 1;
	if (jm1 < 1) {
	    goto L20;
	}
	i__2 = jm1;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    z__[i__] += (r__1 = a[i__ + j * a_dim1], dabs(r__1));
/* L10: */
	}
L20:
/* L30: */
	;
    }
    anorm = 0.f;
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
/* Computing MAX */
	r__1 = anorm, r__2 = z__[j];
	anorm = dmax(r__1,r__2);
/* L40: */
    }

/*     factor */

    spofa_(&a[a_offset], lda, n, info);
    if (*info != 0) {
	goto L180;
    }

/*        rcond = 1/(norm(a)*(estimate of norm(inverse(a)))) . */
/*        estimate = norm(z)/norm(y) where  a*z = y  and  a*y = e . */
/*        the components of  e  are chosen to cause maximum local */
/*        growth in the elements of w  where  trans(r)*w = e . */
/*        the vectors are frequently rescaled to avoid overflow. */

/*        solve trans(r)*w = e */

    ek = 1.f;
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	z__[j] = 0.f;
/* L50: */
    }
    i__1 = *n;
    for (k = 1; k <= i__1; ++k) {
	if (z__[k] != 0.f) {
	    r__1 = -z__[k];
	    ek = r_sign(&ek, &r__1);
	}
	if ((r__1 = ek - z__[k], dabs(r__1)) <= a[k + k * a_dim1]) {
	    goto L60;
	}
	s = a[k + k * a_dim1] / (r__1 = ek - z__[k], dabs(r__1));
	sscal_(n, &s, &z__[1], &c__1);
	ek = s * ek;
L60:
	wk = ek - z__[k];
	wkm = -ek - z__[k];
	s = dabs(wk);
	sm = dabs(wkm);
	wk /= a[k + k * a_dim1];
	wkm /= a[k + k * a_dim1];
	kp1 = k + 1;
	if (kp1 > *n) {
	    goto L100;
	}
	i__2 = *n;
	for (j = kp1; j <= i__2; ++j) {
	    sm += (r__1 = z__[j] + wkm * a[k + j * a_dim1], dabs(r__1));
	    z__[j] += wk * a[k + j * a_dim1];
	    s += (r__1 = z__[j], dabs(r__1));
/* L70: */
	}
	if (s >= sm) {
	    goto L90;
	}
	t = wkm - wk;
	wk = wkm;
	i__2 = *n;
	for (j = kp1; j <= i__2; ++j) {
	    z__[j] += t * a[k + j * a_dim1];
/* L80: */
	}
L90:
L100:
	z__[k] = wk;
/* L110: */
    }
    s = 1.f / sasum_(n, &z__[1], &c__1);
    sscal_(n, &s, &z__[1], &c__1);

/*        solve r*y = w */

    i__1 = *n;
    for (kb = 1; kb <= i__1; ++kb) {
	k = *n + 1 - kb;
	if ((r__1 = z__[k], dabs(r__1)) <= a[k + k * a_dim1]) {
	    goto L120;
	}
	s = a[k + k * a_dim1] / (r__1 = z__[k], dabs(r__1));
	sscal_(n, &s, &z__[1], &c__1);
L120:
	z__[k] /= a[k + k * a_dim1];
	t = -z__[k];
	i__2 = k - 1;
	saxpy_(&i__2, &t, &a[k * a_dim1 + 1], &c__1, &z__[1], &c__1);
/* L130: */
    }
    s = 1.f / sasum_(n, &z__[1], &c__1);
    sscal_(n, &s, &z__[1], &c__1);

    ynorm = 1.f;

/*        solve trans(r)*v = y */

    i__1 = *n;
    for (k = 1; k <= i__1; ++k) {
	i__2 = k - 1;
	z__[k] -= sdot_(&i__2, &a[k * a_dim1 + 1], &c__1, &z__[1], &c__1);
	if ((r__1 = z__[k], dabs(r__1)) <= a[k + k * a_dim1]) {
	    goto L140;
	}
	s = a[k + k * a_dim1] / (r__1 = z__[k], dabs(r__1));
	sscal_(n, &s, &z__[1], &c__1);
	ynorm = s * ynorm;
L140:
	z__[k] /= a[k + k * a_dim1];
/* L150: */
    }
    s = 1.f / sasum_(n, &z__[1], &c__1);
    sscal_(n, &s, &z__[1], &c__1);
    ynorm = s * ynorm;

/*        solve r*z = v */

    i__1 = *n;
    for (kb = 1; kb <= i__1; ++kb) {
	k = *n + 1 - kb;
	if ((r__1 = z__[k], dabs(r__1)) <= a[k + k * a_dim1]) {
	    goto L160;
	}
	s = a[k + k * a_dim1] / (r__1 = z__[k], dabs(r__1));
	sscal_(n, &s, &z__[1], &c__1);
	ynorm = s * ynorm;
L160:
	z__[k] /= a[k + k * a_dim1];
	t = -z__[k];
	i__2 = k - 1;
	saxpy_(&i__2, &t, &a[k * a_dim1 + 1], &c__1, &z__[1], &c__1);
/* L170: */
    }
/*        make znorm = 1.0 */
    s = 1.f / sasum_(n, &z__[1], &c__1);
    sscal_(n, &s, &z__[1], &c__1);
    ynorm = s * ynorm;

    if (anorm != 0.f) {
	*rcond = ynorm / anorm;
    }
    if (anorm == 0.f) {
	*rcond = 0.f;
    }
L180:
    return 0;
} /* spoco_ */

#ifdef __cplusplus
}
#endif
