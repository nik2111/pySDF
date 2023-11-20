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
        assert m < self.n, "spin number should be less than number of spins"
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

        assert m < self.n, "spin number should be less than number of spins"
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
        
        assert m < self.n, "spin number should be less than number of spins"
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

        assert m < self.n, "spin number should be less than number of spins"
        return qt.tensor( [qt.sigmay() if i==m else qt.qeye(2) for i in range(self.n)] )
    
    def X(self):
        """
        Sum of sigmaX operators for all spins
        Returns
        -------
        qutip.Qobj
            Sum of sigmaX operators for all spins
        """
        return sum([self.sigmaX(i) for i in range(self.n)])
    
    def Y(self):
        """
        Sum of sigmaY operators for all spins
        Returns
        -------
        qutip.Qobj
            Sum of sigmaY operators for all spins
        """
        return sum([self.sigmaY(i) for i in range(self.n)])
    
    def Z(self):
        """
        Sum of sigmaZ operators for all spins
        Returns
        -------
        qutip.Qobj
            Sum of sigmaZ operators for all spins
        """
        return sum([self.sigmaZ(i) for i in range(self.n)])
    
    def sigmaPhiSum(self,phi):
        """
        Sum of sigmaPhi operators for all spins
        Parameters
        ----------
        phi : float
            angle in radians
        Returns
        -------
        qutip.Qobj
            Sum of sigmaPhi operators for all spins
        """
        return sum([self.sigmaPhi(phi,i) for i in range(self.n)])
    
class SpinStates(object):
    """
    class for creating spin states for multiple spins
    Parameters
    ----------
    n : int, optional (default=2)
        number of spins
    """
    def __init__(self,n=2) -> None:
        self.n = n
        self._Z0 = qt.basis(2,0)
        self._Z1 = qt.basis(2,1)
        self._X0 = (self._Z0 + self._Z1)/np.sqrt(2)
        self._X1 = (self._Z0 - self._Z1)/np.sqrt(2)
        self._Y0 = (self._Z0 + 1j*self._Z1)/np.sqrt(2)
        self._Y1 = (self._Z0 - 1j*self._Z1)/np.sqrt(2)

    @property
    def Z0all(self):
        """
        returns all spins in 0 state in the Z basis
        Returns
        -------
        qutip.Qobj
            all spins in 0 state in the Z basis
        """
        if self.n == 1:
            return self._Z0
        else:
            return qt.tensor([self._Z0 for i in range(self.n)])

    @property    
    def Z1all(self):
        """
        returns all spins in 1 state in the Z basis
        Returns
        -------
        qutip.Qobj
            all spins in 1 state in the Z basis
        """
        if self.n == 1:
            return self._Z1
        else:
            return qt.tensor([self._Z1 for i in range(self.n)])
        
    @property
    def X0all(self):
        """
        returns all spins in 0 state in the X basis
        Returns
        -------
        qutip.Qobj
            all spins in 0 state in the X basis
        """
        if self.n == 1:
            return self._X0
        else:
            return qt.tensor([self._X0 for i in range(self.n)])
        
    @property
    def X1all(self):
        """
        returns all spins in 1 state in the X basis
        Returns
        -------
        qutip.Qobj
            all spins in 1 state in the X basis
        """
        if self.n == 1:
            return self._X1
        else:
            return qt.tensor([self._X1 for i in range(self.n)])
        
    @property
    def Y0all(self):
        """
        returns all spins in 0 state in the Y basis
        Returns
        -------
        qutip.Qobj
            all spins in 0 state in the Y basis
        """
        if self.n == 1:
            return self._Y0
        else:
            return qt.tensor([self._Y0 for i in range(self.n)])
        
    @property
    def Y1all(self):
        """
        returns all spins in 1 state in the Y basis
        Returns
        -------
        qutip.Qobj
            all spins in 1 state in the Y basis
        """
        if self.n == 1:
            return self._Y1
        else:
            return qt.tensor([self._Y1 for i in range(self.n)])
        
    def Zarb(self,str_rep):
        """
        returrns spins in z basis according to the string representation
        Parameters
        ----------
        str_rep : str
            string representation of the state
        Returns
        -------
        qutip.Qobj
            spins in z basis according to the string representation
        """
        if len(str_rep) != self.n:
            raise ValueError("length of string representation should be equal to number of spins")
        
        for i in str_rep:
            if i not in ['0','1']:
                raise ValueError("string representation should contain only 0 and 1")
        
        return qt.tensor([self._Z0 if i=='0' else self._Z1 for i in str_rep])
    
    def Xarb(self,str_rep):
        """
        returrns spins in x basis according to the string representation
        Parameters
        ----------
        str_rep : str
            string representation of the state
        Returns
        -------
        qutip.Qobj
            spins in x basis according to the string representation
        """
        if len(str_rep) != self.n:
            raise ValueError("length of string representation should be equal to number of spins")
        
        for i in str_rep:
            if i not in ['0','1']:
                raise ValueError("string representation should contain only 0 and 1")
            
        return qt.tensor([self._X0 if i=='0' else self._X1 for i in str_rep])
    
    def Yarb(self,str_rep):
        """
        returrns spins in y basis according to the string representation
        Parameters
        ----------
        str_rep : str
            string representation of the state
        Returns
        -------
        qutip.Qobj
            spins in y basis according to the string representation
        """
        if len(str_rep) != self.n:
            raise ValueError("length of string representation should be equal to number of spins")
        
        for i in str_rep:
            if i not in ['0','1']:
                raise ValueError("string representation should contain only 0 and 1")
            
        return qt.tensor([self._Y0 if i=='0' else self._Y1 for i in str_rep])

    

    