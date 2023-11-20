import pySDF
import numpy as np
import qutip as qt


def test_sigmaPhi():
    """
    Tests if the sigmaPhi operator is created correctly for 2 and 3 spins
    """
    
    phi = np.random.rand() * 2 * np.pi
    sigmaPhiSingle =  qt.sigmax()*np.cos(phi) + qt.sigmay()*np.sin(phi)

    spins2 = pySDF.SpinOperators(2)
    sigmaPhi0 = qt.tensor([sigmaPhiSingle,qt.qeye(2)])
    sigmaPhi1 = qt.tensor([qt.qeye(2),sigmaPhiSingle])
    assert (spins2.sigmaPhi(phi,0) - sigmaPhi0).norm() < 1e-10
    assert (spins2.sigmaPhi(phi,1) - sigmaPhi1).norm() < 1e-10
    
    
    spins3 = pySDF.SpinOperators(3)
    sigmaPhi0 = qt.tensor([sigmaPhiSingle,qt.qeye(2),qt.qeye(2)])
    sigmaPhi1 = qt.tensor([qt.qeye(2),sigmaPhiSingle,qt.qeye(2)])
    sigmaPhi2 = qt.tensor([qt.qeye(2),qt.qeye(2),sigmaPhiSingle])

    assert (spins3.sigmaPhi(phi,0) - sigmaPhi0).norm() < 1e-10
    assert (spins3.sigmaPhi(phi,1) - sigmaPhi1).norm() < 1e-10
    assert (spins3.sigmaPhi(phi,2) - sigmaPhi2).norm() < 1e-10

def test_sigmaZ():
    """
    Tests if the sigmaZ operator is created correctly for 2 and 3 spins
    """
    spins2 = pySDF.SpinOperators(2)
    sigmaZ0 = qt.tensor([qt.sigmaz(),qt.qeye(2)])
    sigmaZ1 = qt.tensor([qt.qeye(2),qt.sigmaz()])
    assert (spins2.sigmaZ(0) - sigmaZ0).norm() < 1e-10
    assert (spins2.sigmaZ(1) - sigmaZ1).norm() < 1e-10

    spins3 = pySDF.SpinOperators(3)
    sigmaZ0 = qt.tensor([qt.sigmaz(),qt.qeye(2),qt.qeye(2)])
    sigmaZ1 = qt.tensor([qt.qeye(2),qt.sigmaz(),qt.qeye(2)])
    sigmaZ2 = qt.tensor([qt.qeye(2),qt.qeye(2),qt.sigmaz()])

    assert (spins3.sigmaZ(0) - sigmaZ0).norm() < 1e-10
    assert (spins3.sigmaZ(1) - sigmaZ1).norm() < 1e-10
    assert (spins3.sigmaZ(2) - sigmaZ2).norm() < 1e-10


def test_sigmaX():
    """
    Tests if the sigmaX operator is created correctly for 2 and 3 spins
    """
    spins2 = pySDF.SpinOperators(2)
    sigmaX0 = qt.tensor([qt.sigmax(),qt.qeye(2)])
    sigmaX1 = qt.tensor([qt.qeye(2),qt.sigmax()])
    assert (spins2.sigmaX(0) - sigmaX0).norm() < 1e-10
    assert (spins2.sigmaX(1) - sigmaX1).norm() < 1e-10

    spins3 = pySDF.SpinOperators(3)
    sigmaX0 = qt.tensor([qt.sigmax(),qt.qeye(2),qt.qeye(2)])
    sigmaX1 = qt.tensor([qt.qeye(2),qt.sigmax(),qt.qeye(2)])
    sigmaX2 = qt.tensor([qt.qeye(2),qt.qeye(2),qt.sigmax()])

    assert (spins3.sigmaX(0) - sigmaX0).norm() < 1e-10
    assert (spins3.sigmaX(1) - sigmaX1).norm() < 1e-10
    assert (spins3.sigmaX(2) - sigmaX2).norm() < 1e-10


def test_sigmaY():
    """
    Tests if the sigmaY operator is created correctly for 2 and 3 spins
    """
    spins2 = pySDF.SpinOperators(2)
    sigmaY0 = qt.tensor([qt.sigmay(),qt.qeye(2)])
    sigmaY1 = qt.tensor([qt.qeye(2),qt.sigmay()])
    assert (spins2.sigmaY(0) - sigmaY0).norm() < 1e-10
    assert (spins2.sigmaY(1) - sigmaY1).norm() < 1e-10

    spins3 = pySDF.SpinOperators(3)
    sigmaY0 = qt.tensor([qt.sigmay(),qt.qeye(2),qt.qeye(2)])
    sigmaY1 = qt.tensor([qt.qeye(2),qt.sigmay(),qt.qeye(2)])
    sigmaY2 = qt.tensor([qt.qeye(2),qt.qeye(2),qt.sigmay()])

    assert (spins3.sigmaY(0) - sigmaY0).norm() < 1e-10
    assert (spins3.sigmaY(1) - sigmaY1).norm() < 1e-10
    assert (spins3.sigmaY(2) - sigmaY2).norm() < 1e-10

def test_X():
    """
    Tests if the X operator is created correctly for 2 and 3 spins
    """

    spins2 = pySDF.SpinOperators(2)
    sigmaX0 = qt.tensor([qt.sigmax(),qt.qeye(2)])
    sigmaX1 = qt.tensor([qt.qeye(2),qt.sigmax()])
    Xsum = sigmaX0 + sigmaX1
    assert (spins2.X() - Xsum).norm() < 1e-10

    spins3 = pySDF.SpinOperators(3)
    sigmaX0 = qt.tensor([qt.sigmax(),qt.qeye(2),qt.qeye(2)])
    sigmaX1 = qt.tensor([qt.qeye(2),qt.sigmax(),qt.qeye(2)])
    sigmaX2 = qt.tensor([qt.qeye(2),qt.qeye(2),qt.sigmax()])
    Xsum = sigmaX0 + sigmaX1 + sigmaX2
    assert (spins3.X() - Xsum).norm() < 1e-10

def test_Y():
    """
    Tests if the Y operator is created correctly for 2 and 3 spins
    """

    spins2 = pySDF.SpinOperators(2)
    sigmaY0 = qt.tensor([qt.sigmay(),qt.qeye(2)])
    sigmaY1 = qt.tensor([qt.qeye(2),qt.sigmay()])
    Ysum = sigmaY0 + sigmaY1
    assert (spins2.Y() - Ysum).norm() < 1e-10

    spins3 = pySDF.SpinOperators(3)
    sigmaY0 = qt.tensor([qt.sigmay(),qt.qeye(2),qt.qeye(2)])
    sigmaY1 = qt.tensor([qt.qeye(2),qt.sigmay(),qt.qeye(2)])
    sigmaY2 = qt.tensor([qt.qeye(2),qt.qeye(2),qt.sigmay()])
    Ysum = sigmaY0 + sigmaY1 + sigmaY2
    assert (spins3.Y() - Ysum).norm() < 1e-10

def test_Z():
    """
    Tests if the Z operator is created correctly for 2 and 3 spins
    """

    spins2 = pySDF.SpinOperators(2)
    sigmaZ0 = qt.tensor([qt.sigmaz(),qt.qeye(2)])
    sigmaZ1 = qt.tensor([qt.qeye(2),qt.sigmaz()])
    Zsum = sigmaZ0 + sigmaZ1
    assert (spins2.Z() - Zsum).norm() < 1e-10

    spins3 = pySDF.SpinOperators(3)
    sigmaZ0 = qt.tensor([qt.sigmaz(),qt.qeye(2),qt.qeye(2)])
    sigmaZ1 = qt.tensor([qt.qeye(2),qt.sigmaz(),qt.qeye(2)])
    sigmaZ2 = qt.tensor([qt.qeye(2),qt.qeye(2),qt.sigmaz()])
    Zsum = sigmaZ0 + sigmaZ1 + sigmaZ2
    assert (spins3.Z() - Zsum).norm() < 1e-10

def test_sigmaPhiSum():
    """
    Tests if the sigmaPhiSum operator is created correctly for 2 and 3 spins
    """
    
    phi = np.random.rand() * 2 * np.pi
    sigmaPhiSingle =  qt.sigmax()*np.cos(phi) + qt.sigmay()*np.sin(phi)

    spins2 = pySDF.SpinOperators(2)
    spins2 = pySDF.SpinOperators(2)
    sigmaPhi0 = qt.tensor([sigmaPhiSingle,qt.qeye(2)])
    sigmaPhi1 = qt.tensor([qt.qeye(2),sigmaPhiSingle])
    sigmaPhiSum = sigmaPhi0 + sigmaPhi1
    assert (spins2.sigmaPhiSum(phi) - sigmaPhiSum).norm() < 1e-10
    
    spins3 = pySDF.SpinOperators(3)
    sigmaPhi0 = qt.tensor([sigmaPhiSingle,qt.qeye(2),qt.qeye(2)])
    sigmaPhi1 = qt.tensor([qt.qeye(2),sigmaPhiSingle,qt.qeye(2)])
    sigmaPhi2 = qt.tensor([qt.qeye(2),qt.qeye(2),sigmaPhiSingle])
    sigmaPhiSum = sigmaPhi0 + sigmaPhi1 + sigmaPhi2
    assert (spins3.sigmaPhiSum(phi) - sigmaPhiSum).norm() < 1e-10


    
    
