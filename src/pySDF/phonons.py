import qutip as qt

class PhononOperators(object):
    '''
    Class for phonon operators for n Harmonic Oscillators
    Parameters
    ----------
    n : int
        number of bosons
    dim : int
        dimension of the Hilbert space for each boson
    
    '''
    def __init__(self,n,dim) -> None:
        self.n = n
        self.dim = dim

    def _a(self):
        '''
        annihilation operator for a single boson
        Returns
        -------
        qutip.Qobj
            annihilation operator for a single boson
        '''
        return qt.destroy(self.dim)
    
    def _adag(self):
        '''
        creation operator for a single boson
        Returns
        -------
        qutip.Qobj
            creation operator for a single boson
        '''
        return qt.create(self.dim)
    
    def a(self,m):
        '''
        annihilation operator for mth boson
        Parameters
        ----------
        m : int
            index of the boson
        Returns
        -------
        qutip.Qobj
            annihilation operator for mth boson
        '''
        assert m < self.n, "boson number should be less than the number of bosons"
        
        if self.n == 1:
            return self._a()
        else:
            return qt.tensor( [self._a() if i==m else qt.qeye(self.dim) for i in range(self.n)] )
        
    def adag(self,m):
        '''
        creation operator for mth boson
        Parameters
        ----------
        m : int
            index of the boson
        Returns
        -------
        qutip.Qobj
            creation operator for mth boson
        '''
        assert m <= self.n, "boson number should be less than number of bosons"
        
        if self.n == 1:
            return self._adag()
        else:
            return qt.tensor( [self._adag() if i==m else qt.qeye(self.dim) for i in range(self.n)] )

    @property   
    def I(self):
        '''
        identity operator for n bosons
        Returns
        -------
        qutip.Qobj
            identity operator for n bosons
        '''
        if self.n == 1:
            return qt.qeye(self.dim)
        else:
            return qt.tensor( [qt.qeye(self.dim) for i in range(self.n)] )
    
class PhononStates:
    '''
    Class for phonon states
    Parameters
    ----------
    n : int
        number of Harmonic Oscillators
    dim : int
        dimension of the Hilbert space for each Harmonic oscillator
    '''
    def __init__(self,n,dim) -> None:
        self.n = n
        self.dim = dim
    
    @property
    def vacuum(self):
        '''
        Returns
        -------
        qutip.Qobj
            vacuum state for n Oscillators
        '''
        if self.n == 1:
            return qt.basis(self.dim,0)
        else:
            return qt.tensor( [qt.basis(self.dim,0) for i in range(self.n)] )
        
        
