from .spins import *
from .phonons import *
import abc
import itertools

class SpinPhononHamiltonian(abc.ABC):
    """
    Base class for spin phonon Hamiltonians
    
    """
    
    def __init__(self,spins : int, oscillators : int ,oscDim : int,
                 trapFreqs : float or np.ndarray, H : qt.Qobj) -> None:
        self.spins = spins
        self.oscilators = oscillators
        self.osc_dim = oscDim
        self.trapFreqs = trapFreqs * 2 * np.pi
        self.H = H

    def __add__(self,other):
        if not isinstance(other,SpinPhononHamiltonian):
            raise ValueError("can only add two SpinPhononHamiltonian objects")
        
        if self.spins != other.spins or self.oscilators != other.oscilators or self.osc_dim != other.osc_dim:
            raise ValueError("cannot add two SpinDepForce objects with different spins or phonons or osc_dim")
        
        if not np.array_equal(self.trapFreqs,other.trapFreqs):
            raise ValueError("cannot add two SpinDepForce objects with different trapFreqs")
        
        H = self.H + other.H
        return SpinPhononHamiltonian(self.spins,self.oscilators,self.osc_dim,self.trapFreqs,H)


    def evolve(self,psi0 : qt.Qobj, t : float) -> qt.Qobj:
        """
        Parameters
        ----------
        psi0 : qutip.Qobj
            initial state
        t : vector
            time
        Returns
        -------
        qutip.Qobj
            evolved state
        """
        opts = qt.Options(nsteps=100000)
        return qt.mesolve(self.H,psi0,t,args={'i':1j},options=opts)

class SpinDepForce(SpinPhononHamiltonian):
    """
    class representing a spin dependent force Hamiltonian
    Parameters
    ----------
    spins : int
        number of spins
    oscillators : int
        number of oscillators (trap motional modes)
    detuning : float
        detuning of the force in units of 2 * pi * Mhz ( do not multiply by 2 * pi)) 
    spinPhase : float
        phase of the spin dependent force
    motionalPhase : float
        phase of the motional part of the Hamiltonian
    eta : np.ndarray
        matrix of shape (spins,phonons) containing the eta parameters
    rabi : np.ndarray
        vector of size spins containing the rabi frequencies at each ion in units of 2 * pi * Mhz ( do not multiply by 2 * pi))
    oscDim : int
        dimension of the oscillator Hilbert space
    trapFreqs : float or vector
        vector of size phonons containing the trap frequencies of the ions

    """
    id_iter = itertools.count()

    def __init__(self,spins : int, oscillators : int ,detuning : float,
                 spinPhase : float, motionalPhase : float,
                 eta : float or np.ndarray ,rabi : float or np.ndarray, 
                 oscDim : int, trapFreqs : float or np.ndarray
                 ) -> None:         

        self.spins = spins
        self.oscilators = oscillators
        self.detuning = detuning * 2 * np.pi
        self.spinPhase = spinPhase
        self.motionalPhase = motionalPhase
        self.eta = eta
        self.rabi = rabi * 2 * np.pi
        self.osc_dim = oscDim
        self.trapFreqs = trapFreqs * 2 * np.pi
        self.id = next(SpinDepForce.id_iter)

        if not isinstance(self.eta,float) and not isinstance(self.eta, np.ndarray):
            raise ValueError("eta should be a float or  matrix of shape (spins,phonons)")
        
        if isinstance(self.eta,np.ndarray) and  self.eta.shape != (self.spins,self.oscilators):
                raise ValueError("eta should be a matrix of shape (spins,phonons)")
        
        if not isinstance(self.rabi, float) and not isinstance(self.rabi, np.ndarray):
            raise ValueError("rabi should be a float or a vector")
        
        if isinstance(self.rabi,np.ndarray) and  self.rabi.size != self.spins:
                raise ValueError("rabi should be a vector of size spins")
        
        if not isinstance(self.osc_dim,int):
            raise ValueError("osc_dim should be an integer")
        
        if not isinstance(self.trapFreqs,float) and not isinstance(self.trapFreqs,np.ndarray):
            raise ValueError("trapFreqs should be a float or a vector")
        
        if isinstance(self.trapFreqs,np.ndarray) and  self.trapFreqs.size != self.oscilators:
            raise ValueError("trapFreqs should be a vector of size phonons")
        
        
        
        if isinstance(self.trapFreqs,float):
            self.trapFreqs = np.array([self.trapFreqs]*self.oscilators)

        self.H = self.Hamiltonian()
        
        

    def __repr__(self) -> str:
        return f"SpinDepForce(id = {self.id},spins = {self.spins},phonons = {self.oscilators},detuning = {self.detuning},spinPhase = {self.spinPhase},motionalPhase = {self.motionalPhase})"
    
    def __str__(self) -> str:
        return self.__repr__()
    
    
    def Hamiltonian(self) -> qt.Qobj:
        """
        Returns
        -------
        qutip.Qobj
            Hamiltonian
        """
        H = []

        spins = SpinOperators(self.spins)
        phonons = PhononOperators(self.oscilators,self.osc_dim)
        # Creating the carrier term
        for i in range(self.spins):
            H_carr = self.rabi[i] * qt.tensor( [spins.sigmaPhi(self.spinPhase - np.pi/2,i), phonons.I] )
            H_time = f" cos( {self.detuning} * t + {self.motionalPhase} ) "
            H.append([H_carr,H_time])

        # Creating the Force terms with negative exponentials
        for i in range(self.spins):
            for j in range(self.oscilators):
                H_force = self.eta[i,j] * self.rabi[i] * qt.tensor( [spins.sigmaPhi(self.spinPhase,i), phonons.a(j) ] ) 
                H_time = f" cos( {self.detuning} * t + {self.motionalPhase} ) * exp(-( i * {self.trapFreqs[j]} * t) ) "
                H.append([H_force,H_time])

        # Creating the Force terms with positive exponentials
        for i in range(self.spins):
            for j in range(self.oscilators):
                H_force = self.eta[i,j] * self.rabi[i] * qt.tensor( [spins.sigmaPhi(self.spinPhase,i), phonons.adag(j) ] ) 
                H_time = f" cos( {self.detuning} * t + {self.motionalPhase} ) * exp( i * {self.trapFreqs[j]} * t) "
                H.append([H_force,H_time])

        return H
    
    
        

    
    





