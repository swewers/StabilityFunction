r"""
Component graphs of plane curves
================================

This module provides routines which *construct* component graphs (in the sense of
:mod:`semistable_model.curves.component_graphs`) from concrete plane curves over
finite fields.

Relation to :mod:`component_graphs`
-----------------------------------

The companion module :mod:`semistable_model.curves.component_graphs` defines the class
:class:`~semistable_model.curves.component_graphs.ComponentGraph`, which is a purely
combinatorial object: an undirected multigraph with loops, together with a geometric
genus attached to each vertex.  The present module is the geometric interface: it
takes a projective plane curve `X` and produces the corresponding
:class:`~semistable_model.curves.component_graphs.ComponentGraph`.

In particular, vertices of the resulting component graph are numbered abstractly;
no curve objects are stored inside the graph.  All geometric computations needed to
construct the graph (factorization into components, determination of singular points
and branches, computation of geometric genera, etc.) are carried out here.

Plane curves with nodes and cusps
---------------------------------

The basic building blocks are the functions :func:`singular_branches`, :func:`is_node`,
and :func:`is_cusp`.  Given a reduced plane curve `X`, we consider its irreducible
components and the branches through the singular points.  A node contributes an edge
between the vertices corresponding to the two branches through the node (a loop if
both branches lie on the same component).  Multiple nodes between the same components
lead to multiple edges.

For applications to stable curves of genus three (and in particular to the stable
reduction of plane quartics), it is important to allow *cusps* in addition to nodes.
A cusp is not a node and hence does not contribute an edge to the core graph.
Instead, it corresponds (after passing to the stable model) to a genus-one tail attached
to the component containing the cusp.

GIT-stable plane quartics
-------------------------

The main entry point is :func:`component_graph_of_GIT_stable_quartic`.  It takes as
input a plane quartic `X` over a finite field which is GIT-stable, i.e. reduced and
having only nodes and cusps as singularities.  The output is the component graph of
the stable curve obtained by smoothing the cusps and attaching genus-one tails.  The
type of tail attached at a cusp is specified by the optional argument ``cusp_data``:
each cusp receives either an elliptic tail (``"e"``) or a pigtail (``"m"``).  If
``cusp_data`` is not given, elliptic tails are attached by default.

Splitting fields and rationality
--------------------------------

Several constructions in this module are most naturally performed after a finite
extension of the ground field over which all components, singular points and branches
become rational.  The helper function :func:`splitting_field` produces such an
extension in the situations considered here, and the remaining routines implicitly
work over this field.

The emphasis of this module is on providing a reliable bridge from explicit plane
curves to the purely combinatorial invariants implemented by
:class:`~semistable_model.curves.component_graphs.ComponentGraph` (canonical signatures,
isomorphism testing, and classification in small genus).
"""

from semistable_model.curves.component_graphs import ComponentGraph
from sage.all import Set, lcm, GF



def component_graph_of_GIT_stable_quartic(X, cusp_data=None, check_degree=True):
    """

    INPUT:
    
    - ``X`` -- a GIT-stable plane quartic over a finite field
    - ``cusp_data`` -- data that provides the resolved tail types for each cusp
    - ``check_degree`` -- a boolean

    The condition on `X` means that `X` is reduced and has at most nodes and cusps
    as singularities. Unless``cusp_data`` is ``None`` we also assume that the cusps
    are rational points of `X`.

    ``cusp_data`` is a list of pairs (`P`, ``tail_type``), where `P` is a cusp,
    as a rational point of `X`, and ``tail_type`` is either "e" (for an elliptic 
    tail), or "m" (for a pigtail).
     

    OUTPUT:
    
    The component graph of the stable curve `X_1` of genus `3`which is obtained 
    by smoothing the cusps and attaching to it a one-tail of the kind indicated
    by ``cusp_data``. 

    Note that the component graph is of type :class:`ComponentGraph` and is
    a purely combinatorial object.

    If cusp_data is not None, it must contain an entry for each cusp of X 
    (otherwise a ValueError is raised). If ``cusp_data`` is ``None`` then 
    we attach to each cusp an elliptic tail.

    If ``check_degree`` is ``False`` it is not checked whether `X` is actually
    a quartic.

    EXAMPLES:

        sage: from semistable_model.curves.plane_curves import ProjectivePlaneCurve
        sage: k = GF(2)
        sage: R.<x,y,z> = k[]
        sage: X = ProjectivePlaneCurve(x*y*z)
        sage: G = component_graph_of_GIT_stable_quartic(X, check_degree=False); G
        abstract component graph of a semistable projective curve

        sage: print(G.as_text())
        Component graph of a semistable curve
        ------------------------------------
        Connected: True
        Total genus: 1

        Vertices:
        v0: genus=0, loops=0, elliptic_tails=0, pigtails=0  [core]
        v1: genus=0, loops=0, elliptic_tails=0, pigtails=0  [core]
        v2: genus=0, loops=0, elliptic_tails=0, pigtails=0  [core]

        Core adjacency (with multiplicities):
        v0 -- v1 : 1
        v0 -- v2 : 1
        v1 -- v2 : 1

        sage: k = GF(3)
        sage: R.<x,y,z> = k[]
        sage: X = ProjectivePlaneCurve((x^2+y^2-z^2)*((x-z)^2+y^2+z^2))
        sage: G = component_graph_of_GIT_stable_quartic(X)
        sage: print(G.as_text())
        Component graph of a semistable curve
        ------------------------------------
        Connected: True
        Total genus: 3

        Vertices:
        v0: genus=0, loops=0, elliptic_tails=0, pigtails=0  [core]
        v1: genus=0, loops=0, elliptic_tails=0, pigtails=0  [core]

        Core adjacency (with multiplicities):
        v0 -- v1 : 4

        sage: k = GF(2)
        sage: R.<x,y,z> = k[]
        sage: X = ProjectivePlaneCurve(x^4 + x*y^2*z + x^2*z^2 + z^4)
        sage: G = component_graph_of_GIT_stable_quartic(X)
        sage: print(G.as_text())
        Component graph of a semistable curve
        ------------------------------------
        Connected: True
        Total genus: 3

        Vertices:
        v0: genus=0, loops=1, elliptic_tails=2, pigtails=0  [core]
        v1: genus=1, loops=0                               [elliptic tail]
        v2: genus=1, loops=0                               [elliptic tail]

        Core adjacency (with multiplicities):
        v0 -- v0 : 1 (loops)

    """
    # checking assumptions
    assert X.defining_polynomial().is_squarefree(), "X must be reduced"
    if check_degree:
        assert X.degree() == 4, "X must be a quartic"
    k = X.base_ring()
    assert k.is_field() and k.is_finite(), "X has to be defined over a finite field"

    # base change to the *splitting field* of X; this guarantees that
    # 1. all componenent are absolutely irreducible
    # 2. all singular points *and* their tangent lines are rational
    # (in fact, 2. implies 1.) 
    k1  = splitting_field(X)
    if not k1 == k:
        X = X.base_change(k1)
        if not cusp_data is None:
            cusp_data = [(P.change_ring(k1), t) for P, t in cusp_data]
        k = k1

    # We recompute the irreducible components and the singular branches
    # They are now all *rational* i.e. defined over the base field. 
    # This allows us to compare centers of different branches; note that
    # we use the method `.rational_center` for this
    components = X.irreducible_components()
    branches = singular_branches(X, components)
    singular_points = Set(b.rational_center() for b in branches)
    # this dictionary remembers all singular branches through a given point 
    bs_at_P = {}
    for b in branches:
        bs_at_P.setdefault(b.rational_center(), []).append(b)

    nodes = []
    cusps = []
    for P in singular_points:
        if is_node(X, P):
            nodes.append(P)
        elif is_cusp(X, P):
            cusps.append(P)
        else:
            raise ValueError("X is not GIT-stable")
        
    if cusp_data is None:
        # by default, one-tails are elliptic tails
        cusp_data = [(P, "e") for P in cusps]
    else:
        # check that each cusp occurs exactly once
        if not all(len([Q for Q, _ in cusp_data if Q == P]) == 1 
                  for P in cusps):
            raise ValueError("every cusp must occur exactly once")
        for Q, _ in cusp_data:
            if not any(Q == P for P in cusps):
                raise ValueError("cusp_data contains a point which is not a cusp of X")

    G = ComponentGraph()
    comp_to_vert = {}
    for Y in components:
        # I can use the components as items because I only 
        # draw them from the initial list `components`
        comp_to_vert[Y] = G.add_vertex(Y.geometric_genus())
    for P in nodes:
        # there must be exactly two branches passing through P 
        e = bs_at_P[P]
        assert len(e) == 2, f"the point {P} is not a node"
        G.add_edge(comp_to_vert[e[0].curve()], comp_to_vert[e[1].curve()])
    for P in cusps:
        b1 = bs_at_P[P][0]
        tail_type = [t for Q, t in cusp_data if Q == P][0]
        G.add_one_tail(comp_to_vert[b1.curve()], tail_type)
    return G


def splitting_field(X):
    r""" Return the splitting field of a plane curve.
    
    INPUT:

    - `X` -- a plane projective curve 

    OUTPUT:

    a finite extension of `k` over which 

    - all irreducible components are absolutely irreducible
    - all singular points are rational points of the ambient space, and
    - all branches at singular points are *rational*, i.e. their residue
      fields are equal to the base field.
    
    Note that the third condition implies the first two! Therefore, to compute 
    a splitting field, it suffices to make all the singular branches rational.
    
    
    EXAMPLES:

    sage: from semistable_model.curves.plane_curves import ProjectivePlaneCurve
    sage: from semistable_model.curves.component_graphs_of_plane_curves import splitting_field
    sage: k = GF(7)
    sage: R.<x,y,z> = k[]
    sage: F = y^2*z + x^3 + x^2*z
    sage: X = ProjectivePlaneCurve(F)
    sage: splitting_field(X)
    Finite Field in z2 of size 7^2

    """
    S = singular_branches(X)
    p = X.base_ring().characteristic()
    n = lcm(s.residue_field().degree() for s in S)
    return GF(p**n)

    
def singular_branches(X, components=None):
    r""" Return the list of all singular branches of a projective plane curve.

    INPUT:

    - ``X`` -- a projective plane curve over a field
    - ``components`` -- the list of irreducible components of `X`, ``None``

    OUTPUT:

    the list of all branches of `X` passing through the singular points.


    EXAMPLES:

    sage: from semistable_model.curves.plane_curves import ProjectivePlaneCurve
    sage: from semistable_model.curves.component_graphs_of_plane_curves import singular_branches
    sage: k = GF(7)
    sage: R.<x,y,z> = k[]
    sage: F = x^2 + y^2
    sage: X = ProjectivePlaneCurve(F)
    sage: singular_branches(X)
    [Place (u) on Projective Plane Curve with defining polynomial x^2 + y^2 over Finite Field of size 7]

    """
    if components is None:
        components = X.irreducible_components()
    S = X.plane_curve.singular_subscheme().reduce()
    ret = []
    for Y in components:
        ret += Y.intersection_branches(S)
    return ret

def is_node(X, P):
    r""" Return whether `P` is an ordinary double point of `X`.
    
    INPUT:

    - `X` -- a projective plane curve over a field
    - `P` -- a rational point on `X`

    OUTPUT:

    whether `P` is an ordinary double point on `X`, i.e.
    whether the tangent cone is a reduced conic (i.e.
    two distinct lines, possibley after a quadratic extension
    of th ground field).
    
    EXAMPLES:

        sage: from semistable_model.curves.plane_curves import ProjectivePlaneCurve
        sage: k = GF(2)
        sage: R.<x,y,z> = k[]
        sage: X = ProjectivePlaneCurve(x*y*z)
        sage: P = X.point([1,0,0])
        sage: is_node(X, P)
        True

    Note that Sage's native method :meth:`is_ordinary_singularity` gives a false result:

        sage: X.plane_curve.is_ordinary_singularity(P)
        False

        sage: P = X.point([1,1,0])
        sage: is_node(X, P)
        False

    A nonsplit ordinary double point counts as a node, too:

        sage: X = ProjectivePlaneCurve(x^2 + x*y + y^2)
        sage: P = X.point([0,0,1])
        sage: is_node(X, P)
        True

    A cusp is not an ordinary double point:

        sage: k = GF(3)
        sage: R.<x,y,z> = k[]
        sage: X = ProjectivePlaneCurve(x^3 + y^2*z)
        sage: P = X.point([0,0,1])
        sage: is_node(X, P)
        False

    """
    assert P in X.plane_curve, "P must be a point on X"
    x, y, z = X.projective_plane.coordinate_ring().gens()
    F = X.defining_polynomial()(x-P[0], y-P[1], z-P[2]).homogeneous_components()
    return not 1 in F.keys() and 2 in F.keys() and F[2].is_squarefree()


def is_cusp(X, P):
    r""" Return whether `P` is a cusp of `X`.
    
    INPUT:

    - ``X`` -- a projective plane curve over a field
    - ``P`` -- a rational point on `X`

    OUTPUT:

    whether `P` is a cusp, i.e. a singularity of type `A_2`.


    EXAMPLES:

        sage: from semistable_model.curves.plane_curves import ProjectivePlaneCurve
        sage: k = GF(2)
        sage: R.<x,y,z> = k[]
        sage: X = ProjectivePlaneCurve(y^2*z + x^3)
        sage: P = X.point([0,0,1])
        sage: is_cusp(X, P)
        True

    """ 
    assert X.defining_polynomial()(list(P)) == 0, "P must be a point on X"
    x, y, z = X.polynomial_ring.gens()
    F = X.defining_polynomial()(x-P[0], y-P[1], z-P[2]).homogeneous_components()
    if 1 in F.keys() or not 2 in F.keys():
        # P is not a double point:
        return False
    F2 = F[2]
    if F2.is_squarefree():
        # P is an ordinary double point
        return False
    # we must test whether P is really a cusp:
    if not 3 in F.keys():
        return False
    L = F2.factor()[0][0]
    return not F[3] in L.parent().ideal(L)

