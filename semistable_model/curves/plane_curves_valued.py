
# ****************************************************************************
#       Copyright (C) 2025 Kletus Stern <sternwork@gmx.de>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#                  https://www.gnu.org/licenses/
# ****************************************************************************

from itertools import product
from sage.all import matrix, identity_matrix, PolynomialRing, GF, QQ
from semistable_model.curves import ProjectivePlaneCurve
from semistable_model.valuations import LinearValuation


class PlaneCurveOverValuedField(ProjectivePlaneCurve):
  r"""
  Construct...

  EXAMPLES::

    sage: from semistable_model.curves.plane_curves_valued import PlaneCurveOverValuedField
    sage: R.<x,y,z> = QQ[]
    sage: F = z*y^2 - x^3 - x*z^2
    sage: Y = PlaneCurveOverValuedField(F, QQ.valuation(2)); Y
    Projective Plane Curve with defining polynomial -x^3 + y^2*z - x*z^2 over Rational Field with 2-adic valuation
  """

  def __init__(self, polynomial, base_ring_valuation):
    r"""
    Construct...

    EXAMPLES::

      sage: from semistable_model.curves.plane_curves_valued import PlaneCurveOverValuedField
      sage: R.<x,y,z> = QQ[]
      sage: F = z*y^2 - x^3 - x*z^2
      sage: Y = PlaneCurveOverValuedField(F, QQ.valuation(2))
      Projective Plane Curve with defining polynomial -x^3 + y^2*z - x*z^2 over Rational Field with 2-adic valuation
    """
    super().__init__(polynomial)
    self._base_ring_valuation = base_ring_valuation

  def __repr__(self):
    return f"Projective Plane Curve with defining polynomial {self.defining_polynomial()} over {self.base_ring()} with {self.base_ring_valuation()}"

  def base_ring_valuation(self):
    return self._base_ring_valuation

  def base_ring(self):
    return self.base_ring_valuation().domain()

  def base_change(self, phi, v_L):
    r"""
    Return the base change of `self` to an extension of the base field

    INPUT:

    - ``phi`` -- an embedding of the base field to a finite extension `L`
    - ``v_L`` -- an extension of the base field valuation to `L`
    """
    new_poly = self.defining_polynomial().map_coefficients(phi, v_L.domain())
    return PlaneCurveOverValuedField(new_poly, v_L)
    
  def degree(self):
    return self.defining_polynomial().degree()

  def git_semistable_model(self, ramification_index=None):
    r"""
    Return a semistable model of `self`.

    EXAMPLES::

      sage: from semistable_model.curves.plane_curves_valued import PlaneCurveOverValuedField
      sage: R.<x,y,z> = QQ[]
      sage: F = y^4 + 2*x^3*z + x*y^2*z + 2*x*z^3
      sage: Y = PlaneCurveOverValuedField(F, QQ.valuation(2))
      sage: X = Y.git_semistable_model()
      sage: X.base_ring()
      Number Field in piK with defining polynomial x^2 + 2

      sage: F = 16*x^4 + y^4 + 8*y^3*z + 16*x*y*z^2 + 4*x*z^3
      sage: Y = PlaneCurveOverValuedField(F, QQ.valuation(2))
      sage: X = Y.git_semistable_model()
      sage: X.base_ring()
      Number Field in piL with defining polynomial x^12 + 2*x^6 + 2

      sage: X.generic_fiber().base_ring() == X.base_ring()
      True
      
      sage: X.has_git_semistable_reduction()
      True
      
      sage: X.special_fiber()
      Projective Plane Curve with defining polynomial x^4 + x^2*y^2 + y*z^3 over Finite Field of size 2

      sage: F = 4*x^4 + 4*x*y^3 + y^4 + 2*x*z^3 + 4*y*z^3 + z^4
      sage: Y = PlaneCurveOverValuedField(F, QQ.valuation(2))
      sage: X = Y.git_semistable_model()
      sage: X.base_ring()
      Number Field in piK with defining polynomial x^4 + 8*x + 4

      sage: X.generic_fiber().base_ring() == X.base_ring()
      True
      
      sage: X.has_git_semistable_reduction()
      True
    """
    from semistable_model.stability import semistable_reduction_field
    from semistable_model.stability import StabilityFunction

    F = self.defining_polynomial()
    v_K = self.base_ring_valuation()
    L = semistable_reduction_field(F, v_K, ramification_index)
    if ramification_index is not None and L is None:
      return None
    
    # L may not be an absolute field, so we have to turn it
    # into one:
    from_K_to_L_abs, v_L = absolute_field(v_K, L)
    X_L = self.base_change(from_K_to_L_abs, v_L)
    phiL = StabilityFunction(X_L.defining_polynomial(), v_L)
    _, b = phiL.global_minimum()
    T = b.move_to_origin().base_change_matrix()
    return PlaneModel(X_L, T)

  def git_semistable_model_with_rational_cusps(self, ramification_index=None):
    r"""
    Return a semistable models of `self` such that all cusps of its
    reduction are rational.

    EXAMPLES::

      sage: from semistable_model.curves.plane_curves_valued import PlaneCurveOverValuedField
      sage: R.<x,y,z> = QQ[]
      sage: F = y^4 + 2*x^3*z + x*y^2*z + 2*x*z^3
      sage: Y = PlaneCurveOverValuedField(F, QQ.valuation(2))
      sage: X = Y.git_semistable_model_with_rational_cusps()
      sage: X.base_ring()
      Number Field in z2 with defining polynomial x^4 - 2*x^3 + x^2 - 6*x + 9
      sage: Xs = X.special_fiber()
      sage: Xs.rational_cusps()
      [Projective flag given by [u1, 1, 1] and u1*x + u1*y + z,
      Projective flag given by [u1 + 1, u1, 1] and u1*x + z]
    """
    X = self.git_semistable_model(ramification_index)
    Xs = X.special_fiber()
    v_L = X.base_ring_valuation()
    d = Xs.splitting_field_of_singular_points().degree()
    from_L_to_L1, v_L1 = unramified_extension(v_L, d)
    return X.base_change(from_L_to_L1, v_L1)
  
  def git_semistable_models_with_e2_x0_cusps(self, ramification_index=None):
    r"""
    Return a list of semistable models such that all cusps of their
    reductions are rational and at least one cusp is in canonical form.

    EXAMPLES::

      sage: from semistable_model.curves.plane_curves_valued import PlaneCurveOverValuedField
      sage: R.<x,y,z> = QQ[]
      sage: F = y^4 + 2*x^3*z + x*y^2*z + 2*x*z^3
      sage: Y = PlaneCurveOverValuedField(F, QQ.valuation(2))
      sage: X1, X2 = Y.git_semistable_models_with_e2_x0_cusps()
      sage: X1.special_fiber().rational_cusps()
      [Projective flag given by [0, 0, 1] and x,
      Projective flag given by [u1, u1, 1] and x + u1*y + z]
      sage: X2.special_fiber().rational_cusps()
      [Projective flag given by [0, 0, 1] and x,
      Projective flag given by [1, u1, 1] and (u1 + 1)*x + y + z]
    """
    X = self.git_semistable_model_with_rational_cusps(ramification_index)
    L = X.base_ring()
    Xs, _, lift = X.special_fiber()
    cusps = Xs.cusps()
    models = []
    for C in cusps:
      T = C.move_to_e2_x0()
      M = T.map_coefficients(lift, L)
      models.append(X.apply_matrix(M))
    return models

  def git_semistable_models_with_e0_x2_cusps(self, ramification_index=None):
    r"""
    Return a dictionary which maps each cusp to a semistable models 
    where  this cusp is in canonical form.

    EXAMPLES::

      sage: from semistable_model.curves.plane_curves_valued import PlaneCurveOverValuedField
      sage: R.<x,y,z> = QQ[]
      sage: F = y^4 + 2*x^3*z + x*y^2*z + 2*x*z^3
      sage: Y = PlaneCurveOverValuedField(F, QQ.valuation(2))
      sage: X1, X2 = Y.git_semistable_models_with_e0_x2_cusps()
      sage: X1.special_fiber().rational_cusps()
      [Projective flag given by [1, 0, 0] and z,
      Projective flag given by [u1 + 1, 1, 1] and x + u1*y + z]
      sage: X2.special_fiber().rational_cusps()
      [Projective flag given by [1, 0, 0] and z,
      Projective flag given by [1, u1, 1] and u1*x + u1*y + z]
    """
    X = self.git_semistable_model_with_rational_cusps(ramification_index)
    L = X.base_ring()
    Xs, _, lift = X.special_fiber()
    cusps = Xs.cusps()
    models = {}
    for C in cusps:
      T = C.move_to_e0_x2()
      M = T.map_coefficients(lift, L)
      models[C] = X.apply_matrix(M)
    return models


class PlaneModel(ProjectivePlaneCurve):
  r"""
  Construct a plane model of a projective plane curve
  over a valued field to the following conditions.

  INPUT:
  - ``generic_fiber`` -- a curve over a valued field.
  - ``base_change_matrix`` -- an invertible matrix.
  """

  def __init__(self, generic_fiber, base_change_matrix):
    r"""
    Construct a plane model of a projective plane curve
    over a valued field.
    """
    if not base_change_matrix.is_invertible():
      raise ValueError("The base change matrix must be invertible.")
    from semistable_model.stability import BTB_Point
    b = BTB_Point(generic_fiber.base_ring_valuation(),
                  base_change_matrix,
                  [0,0,0])
    F = b.hypersurface_model(generic_fiber.defining_polynomial())
    self._bruhat_tits_building_point = b
    super().__init__(F)
    self._generic_fiber = generic_fiber
    self._make_special_fiber()

  def __repr__(self):
    return f"Plane Model of {self.generic_fiber()}"

  def generic_fiber(self):
    return self._generic_fiber

  def base_ring_valuation(self):
    return self.generic_fiber().base_ring_valuation()

  def as_point_on_BTB(self):
    return self._bruhat_tits_building_point

  def base_change_matrix(self):
    return self.as_point_on_BTB().base_change_matrix()

  def adapted_basis(self):
    return self.as_point_on_BTB().linear_valuation().adapted_basis()

  def base_change(self, phi, v_L):
    r"""
    Return the base change of `self` to an extension of the base field

    INPUT:

    - ``phi`` -- an embedding of the base field to a finite extension `L`
    - ``v_L`` -- an extension of the base field valuation to `L`
    """
    X_L = self.generic_fiber().base_change(phi, v_L)
    new_matrix = self.base_change_matrix().map_coefficients(phi, v_L.domain())
    return PlaneModel(X_L, new_matrix)
    
  def apply_matrix(self, T):
    r"""
    Return the plane model of the generic fiber of `self` with
    base change matrix given by `T * self.base_change_matrix()`.

    EXAMPLES::

      sage: from semistable_model.curves.plane_curves_valued import PlaneCurveOverValuedField, PlaneModel
      sage: R.<x,y,z> = QQ[]
      sage: F = 16*x^4 + y^4 + 8*y^3*z + 16*x*y*z^2 + 4*x*z^3
      sage: Y = PlaneCurveOverValuedField(F, QQ.valuation(2))
      sage: X = PlaneModel(Y, identity_matrix(QQ, 3))
      sage: X.base_change_matrix()
      [1 0 0]
      [0 1 0]
      [0 0 1]
      sage: T = matrix(QQ, [[1,0,0],[1,1,0],[0,0,1]])
      sage: T_X = X.apply_matrix(T)
      sage: T_X.base_change_matrix()
      [1 0 0]
      [1 1 0]
      [0 0 1]
      sage: M = matrix(QQ, [[1,0,0],[0,1,0],[0,1,1]])
      sage: MT_X = T_X.apply_matrix(M)
      sage: MT_X.base_change_matrix()
      [1 0 0]
      [1 1 0]
      [1 1 1]
    This method defines a left action.
      sage: MT_X.base_change_matrix() == M*T
      True
      sage: MT_X.base_change_matrix() == T*M
      False
    """
    return PlaneModel(self.generic_fiber(), T * self.base_change_matrix())

  def special_fiber(self):
    r"""
    Return the special fiber of `self`.

    """
    return self._special_fiber
  
  def _make_special_fiber(self):
    """

    a tripel X, red, lift, where X is the residue field of self,
    red is the reduction map from the base field of self to the base field
    of X, and lift is a section of red.
    """
    v_K = self.base_ring_valuation()
    k = v_K.residue_field()
    k1 = GF(k.cardinality())
    phi = k.an_embedding(k1)
    phi_inv = phi.inverse()
    red0 = v_K.reduce
    red = lambda a: phi(red0(a))
    lift0 = v_K.lift
    lift = lambda a: lift0(phi_inv(a))
    f = self.defining_polynomial().map_coefficients(red, k1)
    Xs = ProjectivePlaneCurve(f)
    self._special_fiber = Xs
    self._red = red
    self._lift = lift 
    
  def has_git_semistable_reduction(self):
    r"""
    Return `True` if the special fiber of `self` is
    semistable and `False` otherwise.
    """
    return self.special_fiber().is_git_semistable()
  
  def cusp_model(self, C):
    r""" Return a plane model with the cusp in normal form.
    
    INPUT:

    - ``C`` - a cups on the special fiber of this model

    OUTPUT:

    an isomorphic integral plane model, where the cusp `C` is in
    normal form. 

    EXAMPLES::

      sage: from semistable_model.curves.plane_curves_valued import PlaneCurveOverValuedField
      sage: R.<x,y,z> = QQ[]
      sage: F = z^2*x^2 - y^3*z + y^4
      sage: X = PlaneCurveOverValuedField(F, QQ.valuation(2))
      sage: XX = X.git_semistable_model()
      sage: Xs = XX.special_fiber()
      sage: C = Xs.cusps()[0]; C
      Projective flag given by [0, 0, 1] and x

      sage: XX.cusp_model(C)
      Plane Model of Projective Plane Curve with defining polynomial y^4 - y^3*z + x^2*z^2 over Rational Field with 2-adic valuation

    """
    T = C.move_to_e0_x2()
    M = T.map_coefficients(self._lift, self.base_ring())
    cusp_model = self.apply_matrix(M)
    # test it
    F = cusp_model.defining_polynomial()
    d = F.degree()
    v_L = self.base_ring_valuation()
    for i in range(3):
      for j in range(3):
        if ((i, j) == (3,0) or (i,j) == (0,2)) and v_L(F[d-i-j, i, j]) != 0:
          raise ValueError(f"the cusp in not in normal form: F = {F}")
        if 2*i+3*j < 6 and v_L(F[d-i-j, i, j]) <= 0:
          raise ValueError("the cusp in not in normal form")
    return cusp_model

  def resolve_cusp(self, C):
    r"""Return the tail data after resolving the cusp `C`.

    INPUT:

    - ``C`` -- a rational cusp on the special fiber of this plane model

    OUTPUT:

    a tuple `(v_L, t, E, e)`, where 
    - `e` is a positive integer,
    - `v_L` is an extension of the base ring valuation to a finite field extension `L` 
      such that the cusp can be resolved over any extension of `L`
      with ramification index `e`, 
    - `t` is a positive rational number, the *thickness* of the node, and 
    - `E` is a semistable plane cubic over the residue field of `L`, 
      the resulting one-tail.  
    

    EXAMPLES::

      sage: from semistable_model.curves.plane_curves_valued import PlaneCurveOverValuedField
      sage: R.<x,y,z> = QQ[]
      sage: F = y^4 + 2*x^3*z + x*y^2*z + 2*x*z^3
      sage: Y = PlaneCurveOverValuedField(F, QQ.valuation(2))
      sage: X = Y.git_semistable_model_with_rational_cusps()
      sage: cusps = X.special_fiber().cusps(); cusps
      [Projective flag given by [0, z2, 1] and x + (z2 + 1)*y + z,
       Projective flag given by [0, z2 + 1, 1] and x + z2*y + z]

      sage: X.resolve_cusp(cusps[0])
      (2-adic valuation,
       1/6,
       Projective Plane Curve over Finite Field in u1 of size 2^2 defined by u1*x^3 + (u1 + 1)*y^2*z + (u1 + 1)*y*z^2,
       3)

    """
    from semistable_model.curves import resolve_cusp
    cusp_model = self.cusp_model(C)
    v_L = self.base_ring_valuation()
    return resolve_cusp(cusp_model.defining_polynomial(), v_L)


# ------------------------------------------------------------------------------

def absolute_field(v_K, L):
  r""" Return an absolute version of a p-adic extension.
  
  INPUT:

  - ``v_K`` -- a p-adic valuation on a number field `K`
  - ``L`` -- a finite field extension

  It is assumed that `v_K` is the unique extension of its restriction
  to `\mathbb{Q}`, and that `v_K` has a unique extension to `L`

  OUTPUT:

  a pair `(v_L,\phi)`, where `v_L` is the unique extension of `v_K` to 
  an absolute number field isomorphic to `L` and `\phi` is the embedding
  of `K` into this field. 

  EXAMPLES:

    sage: from semistable_model.curves.plane_curves_valued import absolute_field
    sage: K = CyclotomicField(3)
    sage: v_K = K.valuation(3)
  
  We test whether the trivial extension works:

    sage: absolute_field(v_K, K)
    

  Now we test a relative extension:

    sage: R.<x> = K[]
    sage: L.<pi> = K.extension(x^3-v_K.uniformizer())
    sage: absolute_field(v_K, L)

  Now test the absolute field:

    sage: L_abs.<pi_abs> = L.absolute_field() 
    sage: absolute_field(v_K, L_abs)

  """
  K = v_K.domain() 
  if K == L:
    phi0 = K.hom(K)
  elif  K == L.base_field():
    phi0 = K.hom(L)
  else:
    embeddings = K.embeddings(L)
    if not embeddings:
      raise ValueError(f"There is no embedding of {K} inot {L}")
    phi0 = embeddings[0]
  if not hasattr(L, "is_absolute") or L.is_absolute():
    L_abs = L
    from_L_to_L_abs = L.hom(L)
  else:
    L_abs = L.absolute_field(L.variable_name()) 
    from_L_to_L_abs = L_abs.structure()[1]
  from_K_to_L_abs = phi0.post_compose(from_L_to_L_abs)
  v_L = L_abs.valuation(v_K.p())
  return from_K_to_L_abs, v_L


def extension_of_valued_field(v_K, L):
  r""" Return an extension of a valued field.
  
  INPUT:

  - ``v_K`` -- a p-adic valuation on a number field `K`
  - ``L`` -- a finite field extension of `K`

  OUTPUT:

  A pair `(phi, v_L)`, where `\phi:K\to L` is an embedding and `v_L` is an 
  extension of `v_K` to `L`. 

  EXAMPLES:

    sage: from semistable_model.curves.plane_curves_valued import extension_of_valued_field
    sage: K = CyclotomicField(3)
    sage: v_K = K.valuation(3)
  
  We test whether the trivial extension works:

    sage: extension_of_valued_field(v_K, K)
    (Identity endomorphism of Cyclotomic Field of order 3 and degree 2,
     3-adic valuation)

  Now we test a relative extension:

    sage: R.<x> = K[]
    sage: L.<pi> = K.extension(x^3-v_K.uniformizer())
    sage: extension_of_valued_field(v_K, L)
    (Coercion map:
       From: Cyclotomic Field of order 3 and degree 2
       To:   Number Field in pi with defining polynomial x^3 - zeta3 - 2 over its base field,
     3-adic valuation)

  Now test the absolute field:

    sage: L_abs.<pi_abs> = L.absolute_field() 
    sage: extension_of_valued_field(v_K, L_abs)
    (Ring morphism:
       From: Cyclotomic Field of order 3 and degree 2
       To:   Number Field in pi_abs with defining polynomial x^6 - 3*x^3 + 3
       Defn: zeta3 |--> -pi_abs^3 + 1,
     3-adic valuation)

  
  """
  K = v_K.domain()
  if K == L:
    return K.hom(K), v_K
  if K == L.base_field():
    return K.hom(L), v_K.extension(L)
  embeddings = K.embeddings(L)
  if not embeddings:
    raise ValueError(f"There is no embedding of {K} inot {L}")
  phi = embeddings[0]
  # the following is a hack; at the moment there is no simple way
  # to construct an extension of a valuation along an arbitrary extension
  v_L = L.valuation(v_K.p())
  return phi, v_L


def unramified_extension(v_K, d):
  r""" Return an unramified extension of a p-adic number field.
  
  INPUT:

  - ``v_K`` -- a p-adic valuation on a number field `K`
  - ``d`` -- a positive integer

  It is assumed that `v_K` is the unique extension of its restriction
  to `\mathbb{Q}`.

  OUTPUT:

  a pair `(phi, v_L)`, where `\phi:K\to L` is an embedding 
  into an extension of degree `d`, unramified and totally inert
  over `v_K`, and `v_L` is the unique extension of `v_K` to `L`.

  The field `L` is realized as an *absolute* number field.
  

  EXAMPLES:

    sage: from semistable_model.curves.plane_curves_valued import unramified_extension
    sage: K = CyclotomicField(3)
    sage: v_K = K.valuation(3)
    sage: unramified_extension(v_K, 3)
    (Composite map:
     From: Cyclotomic Field of order 3 and degree 2
     To:   Number Field in z3 with defining polynomial x^6 - 3*x^5 + 10*x^4 - 13*x^3 + 13*x^2 - 8*x + 4
     Defn:   Coercion map:
             From: Cyclotomic Field of order 3 and degree 2
             To:   Number Field in z3 with defining polynomial x^3 + 2*x + 1 over its base field
           then
             Isomorphism map:
             From: Number Field in z3 with defining polynomial x^3 + 2*x + 1 over its base field
             To:   Number Field in z3 with defining polynomial x^6 - 3*x^5 + 10*x^4 - 13*x^3 + 13*x^2 - 8*x + 4,
     3-adic valuation)

  """
  K = v_K.domain()
  k = v_K.residue_field()
  Rb = PolynomialRing(k, "x")
  fb = Rb.irreducible_element(d)
  f = fb.map_coefficients(v_K.lift, K)
  L_rel = K.extension(f, "z"+str(d))
  L = L_rel.absolute_field("z"+str(d))
  from_L_rel_to_L = L.structure()[1]
  from_K_to_L = K.hom(L_rel).post_compose(from_L_rel_to_L)
  v_L = L.valuation(v_K.p())
  return from_K_to_L, v_L
