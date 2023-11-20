import pySDF
import qutip as qt
import numpy as np

def test_a():
    """
    Tests if the annihilation operator is created correctly for 2 and 3 bosons
    """
    bosons2 = pySDF.PhononOperators(2,2)
    a0 = qt.tensor([qt.destroy(2),qt.qeye(2)])
    a1 = qt.tensor([qt.qeye(2),qt.destroy(2)])
    assert (bosons2.a(0) - a0).norm() < 1e-10
    assert (bosons2.a(1) - a1).norm() < 1e-10

    bosons3 = pySDF.PhononOperators(3,2)
    a0 = qt.tensor([qt.destroy(2),qt.qeye(2),qt.qeye(2)])
    a1 = qt.tensor([qt.qeye(2),qt.destroy(2),qt.qeye(2)])
    a2 = qt.tensor([qt.qeye(2),qt.qeye(2),qt.destroy(2)])

    assert (bosons3.a(0) - a0).norm() < 1e-10
    assert (bosons3.a(1) - a1).norm() < 1e-10
    assert (bosons3.a(2) - a2).norm() < 1e-10


def test_adag():
    """
    Tests if the creation operator is created correctly for 2 and 3 bosons
    """
    bosons2 = pySDF.PhononOperators(2,2)
    adag0 = qt.tensor([qt.create(2),qt.qeye(2)])
    adag1 = qt.tensor([qt.qeye(2),qt.create(2)])
    assert (bosons2.adag(0) - adag0).norm() < 1e-10
    assert (bosons2.adag(1) - adag1).norm() < 1e-10

    bosons3 = pySDF.PhononOperators(3,2)
    adag0 = qt.tensor([qt.create(2),qt.qeye(2),qt.qeye(2)])
    adag1 = qt.tensor([qt.qeye(2),qt.create(2),qt.qeye(2)])
    adag2 = qt.tensor([qt.qeye(2),qt.qeye(2),qt.create(2)])

    assert (bosons3.adag(0) - adag0).norm() < 1e-10
    assert (bosons3.adag(1) - adag1).norm() < 1e-10
    assert (bosons3.adag(2) - adag2).norm() < 1e-10