from .spins import *
from .phonons import *
import abc
import itertools
from typing import Union

class SpinPhononHamiltonian(abc.ABC):
    """
    Base class for spin phonon Hamiltonians
    
    """
    
    def __init__(self,spins : int, oscillators : int ,oscDim : int,
                 trapFreqs : Union[float, np.ndarray], H : qt.Qobj,args : dict,
                 hType : str, tag : str) -> None:
        self.spins = spins
        self.oscilators = oscillators
        self.osc_dim = oscDim
        self.trapFreqs = trapFreqs
        self.H = H
        self.args = args
        self.hType = hType
        self.tag = tag

    def __add__(self,other):
        if not isinstance(other,SpinPhononHamiltonian):
            raise ValueError("can only add two SpinPhononHamiltonian objects")
        
        if self.spins != other.spins or self.oscilators != other.oscilators or self.osc_dim != other.osc_dim:
            raise ValueError("cannot add two SpinPhononHamiltonian objects with different spins or phonons or osc_dim")
        
        if not np.array_equal(self.trapFreqs,other.trapFreqs):
            raise ValueError("cannot add two SpinPhononHamiltonian objects with different trapFreqs")
        
        H = self.H + other.H
        args = self.args | other.args
        hType = self.hType + "+" + other.hType
        id = self.tag + "_" + other.tag
        return SpinPhononHamiltonian(self.spins,self.oscilators,self.osc_dim,self.trapFreqs,H,args,hType,id)


    def evolve(self,psi0 : qt.Qobj, t : float, opts : qt.Options = None) -> qt.Qobj:
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
        if opts is None:
            opts = qt.Options(nsteps=100000)
        
        return qt.mesolve(self.H,psi0,t,args=self.args,options=opts)

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
        detuning of the force in units of 2 * pi * Mhz ( you need to multiply by 2 * pi)) 
    spinPhase : float
        phase of the spin dependent force
    motionalPhase : float
        phase of the motional part of the Hamiltonian
    eta : np.ndarray
        matrix of shape (spins,phonons) containing the eta parameters
    rabi : np.ndarray
        vector of size spins containing the rabi frequencies at each ion in units of angular frequency in Mhz ( you need to multiply by 2 * pi))
    oscDim : int
        dimension of the oscillator Hilbert space
    trapFreqs : float or vector
        vector of size phonons containing the trap frequencies of the ions
    tag : str, optional
        unique identifier for the Hamiltonian. If not provided, a unique id will be generated based on an internal counter.
    hDef : str, optional
        Hamiltonian definition as per https://qutip.org/docs/latest/guide/dynamics/dynamics-time.html.
        If 'str', the Hamiltonian will be defined as a string. If 'func', the Hamiltonian will be defined as a function. The default is 'str'.

    """
    _idIter = itertools.count()

    def __init__(self,spins : int, oscillators : int ,detuning : float,
                 spinPhase : float, motionalPhase : float,
                 eta : Union[float, np.ndarray] ,rabi : Union[float, np.ndarray], 
                 oscDim : int, trapFreqs : Union[float, np.ndarray],
                 tag : str = None, hDef : str = 'str', carrier : bool = True) -> None:         

        self.spins = spins
        self.oscilators = oscillators
        self.detuning = detuning
        self.spinPhase = spinPhase
        self.motionalPhase = motionalPhase
        self.eta = eta
        self.rabi = rabi
        self.osc_dim = oscDim
        self.trapFreqs = trapFreqs
        self.hDef = hDef # determines if time dependent Hamiltonian is defined as a string or a function 
        #see qutip time dependent Hamiltonian definition for more details
        self.carrier = carrier
        
        if tag is not None:
            self.tag = tag
        else:
            self.tag = str( next(SpinDepForce._idIter) )
        self.hType = "SDF" # TODO: Implement a centalized typing system for Hamiltonians

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
        
        if self.hDef not in ['str','func']:
            raise ValueError("hDef should be either \'str\' or \'func\'")

        self.H = self.Hamiltonian()
        self.args = self.generateArgs()
        
        

    def __repr__(self) -> str:
        return f"SpinDepForce(id = {self.tag},spins = {self.spins},phonons = {self.oscilators},detuning = {self.detuning},spinPhase = {self.spinPhase},motionalPhase = {self.motionalPhase})"
    
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
        
        if self.hDef == 'str':
            # Creating the carrier term
            if self.carrier:
                for i in range(self.spins):
                    H_carr = qt.tensor( [spins.sigmaPhi(self.spinPhase - np.pi/2,i), phonons.I] )
                    H_time = f" rabi{i}_SDF{self.tag} * cos( detuning_SDF{self.tag} * t + {self.motionalPhase} ) "
                    H.append([H_carr,H_time])

            # Creating the Force terms with negative exponentials
            for i in range(self.spins):
                for j in range(self.oscilators):
                    H_force = self.eta[i,j] * qt.tensor( [spins.sigmaPhi(self.spinPhase,i), phonons.a(j) ] ) 
                    H_time = f" rabi{i}_SDF{self.tag} * cos( detuning_SDF{self.tag} * t + {self.motionalPhase} ) * exp(-( i * {self.trapFreqs[j]} * t) ) "
                    H.append([H_force,H_time])

            # Creating the Force terms with positive exponentials
            for i in range(self.spins):
                for j in range(self.oscilators):
                    H_force = self.eta[i,j] * qt.tensor( [spins.sigmaPhi(self.spinPhase,i), phonons.adag(j) ] ) 
                    H_time = f" rabi{i}_SDF{self.tag} * cos( detuning_SDF{self.tag} * t + {self.motionalPhase} ) * exp( i * {self.trapFreqs[j]} * t) "
                    H.append([H_force,H_time])
        
        elif self.hDef == 'func':
            # Creating the carrier term
            if self.carrier:
                for i in range(self.spins):
                    H_carr = qt.tensor( [spins.sigmaPhi(self.spinPhase - np.pi/2,i), phonons.I] )
                    H_time = lambda t, args: args[f"rabi{i}_SDF{self.tag}"] * np.cos( args[f"detuning_SDF{self.tag}"] * t + self.motionalPhase )
                    H.append([H_carr,H_time])

            # Creating the Force terms with negative exponentials
            for i in range(self.spins):
                for j in range(self.oscilators):
                    H_force = self.eta[i,j] * qt.tensor( [spins.sigmaPhi(self.spinPhase,i), phonons.a(j) ] ) 
                    H_time = lambda t, args: args[f"rabi{i}_SDF{self.tag}"] * np.cos( args[f"detuning_SDF{self.tag}"] * t + self.motionalPhase ) * np.exp( -1j * self.trapFreqs[j] * t )
                    H.append([H_force,H_time])

            # Creating the Force terms with positive exponentials
            for i in range(self.spins):
                for j in range(self.oscilators):
                    H_force = self.eta[i,j] * qt.tensor( [spins.sigmaPhi(self.spinPhase,i), phonons.adag(j) ] ) 
                    H_time = lambda t, args: args[f"rabi{i}_SDF{self.tag}"] * np.cos( args[f"detuning_SDF{self.tag}"] * t + self.motionalPhase ) * np.exp( 1j * self.trapFreqs[j] * t )
                    H.append([H_force,H_time])

        return H
    
    def generateArgs(self) -> dict:
        """
        Returns
        -------
        dict
            dictionary containing the arguments of the Hamiltonian
        """
        args = {'i':1j}
        for i in range(self.spins):
            args[f"rabi{i}_SDF{self.tag}"] = self.rabi[i]
        args[f"detuning_SDF{self.tag}"] = self.detuning
        return args