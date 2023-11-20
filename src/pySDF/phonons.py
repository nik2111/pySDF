import qutip as qt

class PhononOperators(object):
    '''
    Class for boson operators for n bosons
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
        assert m <= self.n, "boson number should be less than the number of bosons"
        
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
    
