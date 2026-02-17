"""
Integral projective plane curves
================================

This module provides classes for *integral* projective plane curves
and their function fields.

It is intended as a refinement layer over ``ProjectivePlaneCurve``,
adding functionality that only makes sense for irreducible curves,
such as:

    - construction of the function field,
    - restriction of homogeneous rational functions to the curve,
    - branches at singular points,
    - interaction with places of the function field.

The goal is to avoid reliance on Sage's native ``Curve`` class
for function field computations (see Sage issue #41643),
while still leveraging Sage's global function field and
place machinery internally.

This is necessary because Sage currently fails for inseparable 
extensions when calling .zeros() due to missing maximal order data
(as the first example below shows).


EXAMPLES:

Function fields must not be constructed as inseparable extensions:

    sage: F0.<u> = FunctionField(GF(2))
    sage: S.<v> = F0[]
    sage: FF.<v> = F0.extension(v^2+u)
    sage: v.zeros()
    ...
    AttributeError: 'FunctionField_polymod_with_category' object has no attribute '_maximal_order_basis'
    
The curve class in this module constructs its function field as a separable extension
of its base field:

    sage: from semistable_model.curves.integral_plane_curves import IntegralProjectivePlaneCurve
    sage: R.<x,y,z> = GF(2)[]
    sage: C = IntegralProjectivePlaneCurve(y^2*z + y*z^2 + x^3); C
    Projective Plane Curve with defining polynomial x^3 + y^2*z + y*z^2 over Finite Field of size 2

    sage: C.function_field()
    Function field in v defined by v^2 + v + u^3

    sage: u = C.function(x/z)
    sage: v = C.function(y/z)
    sage: v^2 + v + u^3
    0

    sage: u.zeros()
    [Place (u, v), Place (u, v + 1)]

    sage: C = IntegralProjectivePlaneCurve(y^2*z + x^3); C
    Projective Plane Curve with defining polynomial x^3 + y^2*z over Finite Field of size 2

    sage: v = C.function(x/z); v
    sage: v.zeros()
    [Place (u, v)]

Given a place on the function field of the curve, we can define the associated branch;
it encodes the (integral) curve *and* the place.

    sage: p = v.zeros()[0]
    sage: br = CurveBranch(C, p); br
    Place (u, 1/u*v^2) on Projective Plane Curve with defining polynomial x^3 + y^2*z ...

    sage: br.curve()
    Projective Plane Curve with defining polynomial x^3 + y^2*z over Finite Field of size 2

    sage: br.place()
    Place (u, 1/u*v^2)

The *center* of the branch is a point on the projective plane:

    sage: P = br.point(); P
    (0 : 0 : 1)

    sage: P.parent()
    Set of rational points of Projective Space of dimension 2 over Finite Field of size 2

The center of a branch need not be a rational point; in any case, a *rational point* 
on the projective plane over the residue field is returned:

    sage: g = C.function((x^2+x*z+z^2)/z^2); g
    v^2 + v + 1

    sage: q = g.zeros()[0]; q 
    Place (u + 1, v^2 + v + 1)

    sage: P = CurveBranch(C, q).point(); P
    (z2 + 1 : 1 : 1)

    sage: P.parent()
    Set of rational points of Projective Space of dimension 2 over Finite Field in z2 of size 2^2

.. todo::

    - implement closed point as centers of branches/places, which are defined over the
      base field of the curve
    - implement :meth:`IntegralProjectivePlaneCurve.lift_function`

"""

from semistable_model.curves.plane_curves import ProjectivePlaneCurve
from sage.all import SageObject, PolynomialRing, FunctionField, lcm, Set


# ----------------------------------------------------------------------
# Integral projective plane curves
# ----------------------------------------------------------------------

class IntegralProjectivePlaneCurve(ProjectivePlaneCurve):
    """
    An integral (irreducible and reduced) projective plane curve.

    This class extends ``ProjectivePlaneCurve`` by adding:

        - construction of the function field,
        - restriction maps from homogeneous rational functions,
        - access to places of the function field.

    It is assumed that the defining polynomial is irreducible
    over the base field.
    """
    def __init__(self, F):
        """
        Initialize an integral projective plane curve.

        INPUT:
  
        - ``F`` -- homogeneous polynomial in K[x_0, x_1, x_2] over a field `K`.
  
        The polynomial `F` must be irreducible. 

        Note:

        For the moment, the base field `K` must be finite. 

        """
        factors = F.factor()
        if not (len(factors) == 1 and factors[0][1] == 1):
            raise ValueError("The defining polynomial F must be irreducible: "
                             + f"F = {F.factor()}")
        K = F.base_ring()
        if not (K.is_field() and K.is_finite()):
             raise ValueError("The base ring of F must be a finite field:"
                             + f"K = {K}")
        super().__init__(F) 
        # initialize function field etc.
        self._create_function_field()

    def function_field(self):
        """
        Return the function field of the curve.

        """
        return self._function_field
    
    def geometric_genus(self):
        r""" Return the geometric genus of this integral curve.
        
        """
        return self.function_field().genus()
    
    def standard_affine_chart_var(self):
        r""" Return the variable x_i which defines the standard affine chart x_i!=0.
        
        """
        return self._affine_chart_var

    def coordinate_functions(self, affine_chart=None):
        r""" Return the coordinate functions of this projective curve.
        
        If `K[x_0,x_1,x_2]` is the coordinate ring of the ambient projective
        plane, and the standard affine chart is `x_i\neq 0`, then 
        we return the tuple of function field elements 
        
        .. MATH::
        
            x_j/x_i, \quad j=0,\ldots,2. 

        You can overwrite this by setting ``affine_chart`` to the desired index `i`.

        If this irreducible curve is the line `x_i=0`, then this will raise
        an error.

        """
        x = self.polynomial_ring.gens()
        if affine_chart is None:
            xi = self.standard_affine_chart_var()
        else:
            xi = x[affine_chart]
            if self._phi(xi) == 0:
                raise ValueError(f"The curve is defined by {xi} = 0, so i={affine_chart} is not allowed.")
        return [self.function(x[j]/xi) for j in range(3)]
    
    def function(self, h):
        r""" Return the restriction of a rational function to this curve.
        
        INPUT:

        - ``h`` -- either
            * an element of the fraction field of K[x0,x1,x2], or
            * a pair ``(num, den)`` of polynomials in K[x0,x1,x2].

        In both cases, ``num/den`` must be homogeneous of degree 0, i.e.
        ``num`` and ``den`` are homogeneous of the same degree.

        OUTPUT:

        the restriction of `h` to this curve `C`, as an element of the 
        function field `K(C)`.
        
        """
        return self._to_function_field(h)
    
    def lift_function(self, f, affine_chart=None):
        r""" Return a lift of a function on this curve.
        
        INPUT:

        - ``f`` -- an element of the function field of this curve
        - ``affine_chart`` -- an integer `i`, `0\leq i\leq 2`, or ``None``

        OUTPUT:

        A rational function `F = G/x_i^d` on the ambient projective plane
        which restrict to `f`. Here `x_i` is the variable defines the 
        standard affine chart `x_i \neq 0`. 

        The fixed choice of `i` can be overwritten by setting ``affine_chart``
        to the desired index. 
        
        """
        raise NotImplementedError

    def place_to_point(self, v):
        r""" Return the point at the place.
        
        INPUT:

        - ``v`` -- a place of the function field of this curve

        OUTPUT:

        The center of ``v``, as a rational point on the projective
        plane over the residue field of `v`.

        """
        pi = v.local_uniformizer()
        _, _, red = v.residue_field()  # the reduction map
        assert pi.valuation(v) == 1
        coordinate_functions = self.coordinate_functions()
        vals = [f.valuation(v) for f in coordinate_functions]
        m = min(vals)
        P = [red(f*pi**(-m)) for f in coordinate_functions]
        k = v.residue_field()[0]
        return self.projective_plane.base_extend(k)(P)

    def branches_at(self, P):
        """
        Return the branches of the curve at a singular point P.

        Parameters
        ----------
        P :
            A closed point (or rational point) of the curve.

        Returns
        -------
        list of CurveBranch

        Notes
        -----
        Each branch should correspond to a place of the function
        field lying over P.
        """
        raise NotImplementedError

    def places_over_point(self, P):
        """
        Return the places lying over a closed point P.

        Parameters
        ----------
        P :
            A closed point of the curve.

        Returns
        -------
        list
            Places of k(C) with center P.
        """
        raise NotImplementedError
    
    def intersection_branches(self, S, assume_reduced=True):
        r""" Return the branches of this curve which lie on S.
        
        INPUT:

        - ``S`` -- a closed subscheme of the plane, not containing this curve

        OUTPUT:

        The list of all branches of this curve whose center lie in `S`

        """
        if not assume_reduced:
            S = S.reduce()    
        x = self.polynomail_ring.gens()
        F = self.defining_polynomial()
        eqns = list(S.defining_polynomials())
        # check that C is not containd in S
        if all(F.divides(G) for G in eqns):
            raise ValueError("C is contained in S")
        # we gather all the places of Y with center in S in a set;
        # this is ok, because places are hashable and have a unique 
        # representation
        places = set()

        for i in range(3):
            xi = x[i]
            #  we add the places with x_0=..=x_{i-1}=0, x_i!=0
            if not xi.divides(F):
                list_of_zero_sets = []
                for G in eqns:
                    g = self.function(G/xi**G.degree())
                    if g != 0:
                        list_of_zero_sets.append(set(g.zeros()))
                if not list_of_zero_sets:
                    raise ValueError("this shouldn't happen!")
                places |= set.intersection(*list_of_zero_sets)
            eqns.append(xi)

        return [CurveBranch(self, v) for v in places]

        
    def vanishes_on(self, G, v):
        r""" Return whether f vanishes on the branch v.
        
        INPUT:

        - `G` -- a homogenous form
        - `v` -- a place on the function field 

        
        """
        raise NotImplementedError()

    def _create_function_field(self, names=('u', 'v')):
        r"""
        This function creates the function field `k(C)=k(u,v)`
        of this irreducible curve; here `u,v` are standard generators, i.e

        - either `k(C) = k(u)` and `v` is some element, or
        - `k(C) = k(u,v)` such that the minimal polynomial of `v` over `k(u)`
          is separable and of degree `>1`.

        The function also stores two ring homomorphisms  

        .. MATH::

            \phi:k[x_0,x_1,x_2]\to L[t]
        
        and 

        .. MATH::

            \psi:k[x_0,x_1,x_2] \to k[x_0,x_1,x_2]

        such that

        1. `\psi` is an an automorphism induced by a permutation of variables
        2. The composition `\phi\circ\psi` is the morphism induced by
                `x_0\mapsto uT,\; x_1\mapsto vT,\; x_2\mapsto T`, 
        where `u, v` are standard generators of the function field `L`

        In particular, `x_i=\psi(x_2)` is the variable that defines the 
        standard affine chart `x_i\neq 0`. 

        """
        # --- get defining polynomial and its coordinate ring ---
        F = self.defining_polynomial()
        x = self.polynomial_ring.gens()
        FF, phi, psi = _ff_from_polynomial(F, names)
        
        # --- store attributes 
        self._function_field = FF
        self._phi = phi
        self._psi = psi
        self._affine_chart_var = psi(x[2])

    def _to_function_field(self, h):
        r"""
        Map a degree-0 homogeneous rational function on P^2 to the function field k(C).

        INPUT:

        - ``h`` -- either
            * an element of the fraction field of K[x0,x1,x2], or
            * a pair ``(num, den)`` of polynomials in K[x0,x1,x2].

        In both cases, ``num/den`` must be homogeneous of degree 0, i.e.
        ``num`` and ``den`` are homogeneous of the same degree.

        OUTPUT:

        the restriction of `h` to the curve `C`, as an element of the 
        function field `K(C)`.

        """
        F = self.defining_polynomial()
        R = F.parent()
        # --- unpack input into (num, den) in the ambient polynomial ring ---
        if isinstance(h, tuple) and len(h) == 2:
            num, den = h
            num = R(num)
            den = R(den)
        else:
            # Assume h is in a fraction field of something coercing from R
            # Try to extract numerator/denominator in a robust way.
            try:
                num = R(h.numerator())
                den = R(h.denominator())
            except Exception:
                # Fallback: coerce h into fraction field of R and retry
                try:
                    Fr = R.fraction_field()
                    hh = Fr(h)
                    num = R(hh.numerator())
                    den = R(hh.denominator())
                except Exception as e:
                    raise TypeError(
                        "Expected a fraction field element or a pair (num, den) "
                        "with num, den in the ambient polynomial ring."
                    ) from e

        if den == 0:
            raise ZeroDivisionError("Denominator is zero on the curve.")

        # --- homogeneity checks (degree-0 condition) ---
        # We require num and den to be homogeneous of same total degree.
        if not num.is_homogeneous() or not den.is_homogeneous():
            raise ValueError("num and den must be homogeneous polynomials.")
        if num != 0 and num.total_degree() != den.total_degree():
            raise ValueError(
                f"num/den must have degree 0: num and den must have the same degree. num = {num}, den = {den}"
            )
        
        d = den.total_degree()
        phi = self._phi
        num_ff = phi(num)[d]
        den_ff = phi(den)[d]
        if den_ff == 0:
            raise ZeroDivisionError(
                f"Denominator restricts to zero on the curve: {num/den} = {num_ff}/{den_ff}"
            )
        return num_ff/den_ff

    def _from_function_field(self, f):
        """
        (Partial) inverse to _to_function_field, returning a degree-0 homogeneous
        fraction in the ambient fraction field.

        """
        FF = self.function_field()
        f = FF(f)
        R = self.polynomial_ring
        x = R.gens()
        psi = self._psi
        
        # write f as quotient of polynomials in u, v
        G0, H0 = _ff_element_to_polynomial_quotient(f)

        # dehomogenize + substitute
        d = max(G0.degree(), H0.degree())
        y = [psi(x[i]) for i in range(3)]
        G = R(y[2]**d * G0(y[0]/y[2], y[1]/y[2]))
        H = R(y[2]**d * H0(y[0]/y[2], y[1]/y[2]))

        return G/H
        
# -----------------------------------------------------------------------
#         
# some helper functions


def _ff_from_polynomial(F, names):
    r""" Helper function for _create_function_field.
    
    INPUT:

    - ``F`` -- an irreducible homogenous form in `k[x_0,x_2,x_2]`

    OUTPUT:

    A triple `(L, \phi, \psi)`, where `L/k` is a function field and 

    .. MATH::

        \phi:k[x_0,x_1,x_2]\to L[t], \quad \psi:k[x_0,x_1,x_2] \to k[x_0,x_1,x_2]

    are ring homomorphism such that the following holds (for some fixed index `i`):

    1. `\psi` is an automorphism induced by a permutation of the variables
    2. The composition `\phi\circ\psi` is the morphism induced by
            `x_0\mapsto uT,\; x_1\mapsto vT,\; x_2\mapsto T`, 
       where `u, v` are standard generators of the function field `L`
    3. The function field `L` is separable over its base field, i.e.
       the minimal polynomial of `v` over `k(u)` is separable. 
    
    """
    R = F.parent()
    k = R.base_ring()
    x = R.gens()
    FF_base = FunctionField(k, names[0])
    u = FF_base.gen()
    S = PolynomialRing(FF_base, names[1])
    v = S.gen()

    for i2 in range(2, -1, -1):
        if not x[i2].divides(F):
            break
    else:
        raise ValueError(f"F = {F} cannot be irreducible")
    i0, i1 = (i for i in range(3) if i != i2)
    swapped = False
    is_separable = False
    while not is_separable:
        psi = R.hom([x[i0], x[i1], x[i2]])
        phi0 = psi.inverse()
        f = phi0(F)(u, v, S.one()).monic()
        is_separable = f.derivative() != 0
        if is_separable:
            break
        elif not swapped:
            i0, i1 = i1, i0
            swapped = True
        else: 
            # for both ordering of i0, i1, f is inseparable
            # if F is homogenous and irreducible, this can never happen
            raise ValueError(f"F cannot be irreducible: F = {F}")    
    
    if f.degree() == 1:
        FF = FF_base
        v = -f[0]/f[1]
    else:
        FF = FF_base.extension(f)
        v = FF.gen()    
    assert f(v) == 0
    R_to_FF = R.hom([u, v, FF.one()])
    phi1 = phi0.post_compose(R_to_FF)
    FFT = PolynomialRing(FF, "T")
    T = FFT.gen()
    phi = R.hom([phi1(x[0])*T, phi1(x[1])*T, phi1(x[2])*T])
    assert phi(F) == 0, f"phi(F) is not zero! F = {F}"
    assert phi(psi(x[0])) == u*T
    assert phi(psi(x[1])) == v*T
    assert phi(psi(x[2])) == T

    return FF, phi, psi


def _ff_element_to_polynomial_quotient(f):
    r""" Write function field element as quotient of polynomials.
    
    INPUT:

    - ``f`` -- an element of a function field `L=k(u,v)`

    OUTPUT:

    a pair of polynomials `G,H` such that `f=G(u,v)/H(u,v)

    The generators `u,v` are chosen as follows:

    - if `L=k(u)` is a rational function field, then `u` is the 
      standard generator
    - otherwise, `L/k(u)´ is the simple extension with standard generator `v`

    """
    FF = f.parent()
    k = FF.base_ring()
    R = PolynomialRing(k, 2, ["u", "v"])
    u, v = R.gens()
    V, _, phi = FF.vector_space()
    F = phi(f)
    den = lcm(F[i].denominator() for i in range(V.dimension()))
    G = sum((den*F[i]).numerator()(u)*v**i for i in range(V.dimension()))
    H = den(u)
    # check the result
    u = FF.base_field().gen()
    v = FF.gen()
    assert f == G(u,v)/H(u,v), f"f = {f}, G = {G}, H = {H}"
    return G, H



# ----------------------------------------------------------------------
# Branch objects
# ----------------------------------------------------------------------

class CurveBranch(SageObject):
    r""" Return a curve branch.

    INPUT:

    - ``Y`` -- an integral curve over a field
    - ``v`` -- a place of the function of `Y`
    
    OUTPUT:

    an object representing the pair `(Y, v)`.

    """
    def __init__(self, Y, v):
        self._curve = Y
        self._place = v
        self._point = Y.place_to_closed_point(v)

    def __repr__(self):
        return f"{self.place()} on {self.curve()}"

    def curve(self):
        return self._curve
    
    def place(self):
        return self._place
    
    def point(self):
        r""" Return the point on the curve corresponding to this branch.
        
        NOTE:

        Currently, a *rational point* on the projective plane over the
        residue field of the branch is returned. 

        It would be desirable to implement two versions, 
        :meth:`rational_point` and :meth_`closed_point`
         
        """
        return self._point
    
    def rational_point(self):
        r""" Return the corresponding rational point of the projective plane.
        
        If the point is not rational, an error is raised.
        """
        # raise NotImplementedError
        # old version:
        assert self.place().residue_field()[0] == self.curve().base_ring()
        return self.curve().projective_plane(self.point())
    
    def residue_field(self):
        return self.place().residue_field()[0]

