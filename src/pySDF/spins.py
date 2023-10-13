import numpy as np
import qutip as qt

class SpinOperators(object):
    """
    class for creating spin systems
    Parameters
    ----------
    n : int, optional (default=2)
        number of spins
    """
    def __init__(self,n=2) -> None:
        self.n = n

    def _sigmaPhi(self,phi):
        """
        returns sigmaPhi operator for a given angle phi
        Parameters
        ----------
        phi : float
            angle in radians
        Returns
        -------
        qutip.Qobj
            sigma_phi = sigma_x * Cos(phi) + sigma_y * Sin(phi)
        """
        return qt.sigmax()*np.cos(phi) + qt.sigmay()*np.sin(phi)
    
    def sigmaPhi(self,phi,m):
        """
        returns sigmaPhi operator for a given angle phi and spin m
        returns I X I X ......sigma_phi...X I X I with sigma_phi at the mth position
        where sigma_phi = sigma_x * Cos(phi) + sigma_y * Sin(phi)
        Parameters
        ----------
        phi : float
            angle in radians
        m : int
            spin number
        Returns
        -------
        qutip.Qobj
            sigma_phi = sigma_x * Cos(phi) + sigma_y * Sin(phi)
        """
        return qt.tensor( [self._sigmaPhi(phi) if i==m else qt.qeye(2) for i in range(self.n)] )
    
    def sigmaZ(self,m):
        """
        returns sigmaZ operator for a given spin m
        returns I X I X ......sigma_z...X I X I with sigma_z at the mth position
        Parameters
        ----------
        m : int
            spin number
        Returns
        -------
        qutip.Qobj
            sigma_z
        """
        return qt.tensor( [qt.sigmaz() if i==m else qt.qeye(2) for i in range(self.n)] )
    
    def sigmaX(self,m):
        """
        returns sigmaX operator for a given spin m
        returns I X I X ......sigma_x...X I X I with sigma_x at the mth position
        Parameters
        ----------
        m : int
            spin number
        Returns
        -------
        qutip.Qobj
            I X I X ......sigma_x...X I X I with sigma_x at the mth position
        """
        return qt.tensor( [qt.sigmax() if i==m else qt.qeye(2) for i in range(self.n)] )
    
    def sigmaY(self,m):
        """
        returns sigmaY operator for a given spin m
        returns I X I X ......sigma_y...X I X I with sigma_y at the mth position
        Parameters
        ----------
        m : int
            spin number
        Returns
        -------
        qutip.Qobj
            I X I X ......sigma_y...X I X I with sigma_y at the mth position
        """
        return qt.tensor( [qt.sigmay() if i==m else qt.qeye(2) for i in range(self.n)] )