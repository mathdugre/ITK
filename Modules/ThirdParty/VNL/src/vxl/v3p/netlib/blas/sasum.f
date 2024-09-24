*> \brief \b SASUM
*
*  =========== DOCUMENTATION ===========
*
* Online html documentation available at
*            http://www.netlib.org/lapack/explore-html/
*
*  Definition:
*  ===========
*
*       REAL FUNCTION SASUM(N,SX,INCX)
*
*       .. Scalar Arguments ..
*       INTEGER INCX,N
*       ..
*       .. Array Arguments ..
*       REAL SX(*)
*       ..
*
*
*> \par Purpose:
*  =============
*>
*> \verbatim
*>
*>    SASUM takes the sum of the absolute values.
*>    uses unrolled loops for increment equal to one.
*> \endverbatim
*
*  Arguments:
*  ==========
*
*> \param[in] N
*> \verbatim
*>          N is INTEGER
*>         number of elements in input vector(s)
*> \endverbatim
*>
*> \param[in] SX
*> \verbatim
*>          SX is REAL array, dimension ( 1 + ( N - 1 )*abs( INCX ) )
*> \endverbatim
*>
*> \param[in] INCX
*> \verbatim
*>          INCX is INTEGER
*>         storage spacing between elements of SX
*> \endverbatim
*
*  Authors:
*  ========
*
*> \author Univ. of Tennessee
*> \author Univ. of California Berkeley
*> \author Univ. of Colorado Denver
*> \author NAG Ltd.
*
*> \ingroup asum
*
*> \par Further Details:
*  =====================
*>
*> \verbatim
*>
*>     jack dongarra, linpack, 3/11/78.
*>     modified 3/93 to return if incx .le. 0.
*>     modified 12/3/93, array(1) declarations changed to array(*)
*> \endverbatim
*>
*  =====================================================================
      REAL function sasum(n,sx,incx)
*
*  -- Reference BLAS level1 routine --
*  -- Reference BLAS is a software package provided by Univ. of Tennessee,    --
*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
*
*     .. Scalar Arguments ..
      INTEGER incx,n
*     ..
*     .. Array Arguments ..
      REAL sx(*)
*     ..
*
*  =====================================================================
*
*     .. Local Scalars ..
      REAL stemp
      INTEGER i,m,mp1,nincx
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC abs,mod
*     ..
      sasum = 0.0e0
      stemp = 0.0e0
      IF (n.LE.0 .OR. incx.LE.0) RETURN
      IF (incx.EQ.1) THEN
*        code for increment equal to 1
*
*
*        clean-up loop
*
         m = mod(n,6)
         IF (m.NE.0) THEN
            DO i = 1,m
               stemp = stemp + abs(sx(i))
            END DO
            IF (n.LT.6) THEN
               sasum = stemp
               RETURN
            END IF
         END IF
         mp1 = m + 1
         DO i = mp1,n,6
            stemp = stemp + abs(sx(i)) + abs(sx(i+1)) +
     $              abs(sx(i+2)) + abs(sx(i+3)) +
     $              abs(sx(i+4)) + abs(sx(i+5))
         END DO
      ELSE
*
*        code for increment not equal to 1
*
         nincx = n*incx
         DO i = 1,nincx,incx
            stemp = stemp + abs(sx(i))
         END DO
      END IF
      sasum = stemp
      RETURN
*
*     End of SASUM
*
      END
