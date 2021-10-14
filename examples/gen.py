import sisl
import os
import numpy as np
from pymatgen.core import Structure, Molecule


def main():
    a = 2.471153
    c = 25
    g = graphene(a, c)

    xv = sisl.get_sile(
        '../../DS5-single-molecules-in-vacuum/f4tcnq/molecule.XV')
    f4tcnq = xv.read_geometry()
    mol = Molecule([a.symbol for a in f4tcnq.atoms], f4tcnq.xyz)

    n = 7
    scmat = np.array([[n, n, 0], [-n, n, 0], [0, 0, 1]], dtype=np.int32)
    g.make_supercell(scmat)

    cm = mol.center_of_mass
    dz = 3.3 / g.lattice.c
    r0 = g.lattice.matrix.dot([.5, 3/n, .5+dz])
    for s in mol:
        c = s.coords - cm + r0
        g.append(s.species, c, coords_are_cartesian=True)

    geom = sisl.Geometry(
        g.cart_coords, [s.specie.symbol for s in g], g.lattice.matrix)
    # geom.write('struct.fdf')
    # geom.write('struct.xsf')

    print(np.array([2/3, 1/3, 0]).dot(scmat.T))
    print(np.array([1/2, 0, 0]).dot(scmat.T))


def graphene(a, c):
    lattice = np.array([
        [np.sqrt(3)/2*a, -1/2*a, 0],
        [np.sqrt(3)/2*a, +1/2*a, 0],
        [0,      0, c]
    ])
    coords = [
        [1/3, 1/3, 1/2],
        [2/3, 2/3, 1/2],
    ]
    atoms = ['C', 'C']
    return Structure(lattice, atoms, coords)


main()
