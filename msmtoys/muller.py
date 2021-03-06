"""Propagating 2D dynamics on the muller potential using OpenMM.

Currently, we just put a harmonic restraint on the z coordinate,
since OpenMM needs to work in 3D. This isn't really a big deal, except
that it affects the meaning of the temperature and kinetic energy. So
take the meaning of those numbers with a grain of salt.
"""

import numpy as np
import simtk.openmm as mm


class MullerForce(mm.CustomExternalForce):
    """OpenMM custom force for propagation on the Muller Potential. Also
    includes pure python evaluation of the potential energy surface so that
    you can do some plotting"""
    aa = [-1, -1, -6.5, 0.7]
    bb = [0, 0, 11, 0.6]
    cc = [-10, -10, -6.5, 0.7]
    AA = [-200, -100, -170, 15]
    XX = [1, 0, -0.5, -1]
    YY = [0, 0.5, 1.5, 1]
    strength = 0.5

    def __init__(self):
        # start with a harmonic restraint on the Z coordinate
        expression = '1000.0 * z^2'
        for j in range(4):
            # add the muller terms for the X and Y
            fmt = dict(aa=self.aa[j], bb=self.bb[j], cc=self.cc[j],
                       AA=self.AA[j], XX=self.XX[j], YY=self.YY[j])
            expression += '''+ {AA}*exp({aa} *(x - {XX})^2 +
                                {bb} * (x - {XX}) * (y - {YY}) +
                                {cc} * (y - {YY})^2)'''.format(**fmt)

        # Include scaling expression
        expression = ("{strength}*(".format(strength=self.strength) +
                      expression + ")")

        super(MullerForce, self).__init__(expression)


    @classmethod
    def potential(cls, x, y):
        "Compute the potential at a given point x,y"
        value = np.zeros_like(x)
        for j in range(4):
            value += cls.AA[j] * np.exp(cls.aa[j] * (x - cls.XX[j]) ** 2 + \
                                        cls.bb[j] * (x - cls.XX[j]) * (
                                            y - cls.YY[j]) + \
                                        cls.cc[j] * (y - cls.YY[j]) ** 2)
        return value

    # xmin, xmax, ymin, ymax
    bounds = [-1.5, 1.2, -0.2, 2.0]
