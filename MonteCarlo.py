import numpy as np

from utils import progress_bar, clear_progress_bar

# TODO turn this into an abstract class

class MonteCarlo:

    '''
    A base class encapsulating core Monte Carlo simulation logic.
    '''

    def __init__(self, S0, K, T, r, sigma, simulations=10_000, time_steps=250, barrier = 0):
        '''
        Parameters
        ----------
        S0 : float
            Initial stock price
        K : float
            Strike price
        T : float
            Time to maturity (in years)
        r : float
            Risk-free interest rate
        sigma : float
            Volatility
        simulations : int
            Number of simulations
        time_steps : int
            Time steps
        '''
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.simulations = simulations
        self.time_steps = time_steps
        self.barrier = barrier
        self.price_paths = None
        


    @property
    def name(self):
        return self.__class__.__name__


    def simulate_paths(self):
        '''
        Description
        -----------
        Generates paths for the underlying asset price (default using Euler scheme).

        Parameters
        ----------
        with_progress_bar : bool
            If True, a progress bar will be displayed in the console.
        '''
        dt = self.T / self.time_steps
        Z = np.random.standard_normal((self.time_steps, self.simulations)) #A matrix consisiting of random numbers
        S = np.zeros((self.time_steps + 1, self.simulations))  # +1 for initial price S0
        S[0] = self.S0 #Initialises the entire first row of the S matrix to the starting stock price
        
        for t in range(1, self.time_steps + 1):
            S[t] = S[t - 1] * np.exp((self.r - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * Z[t - 1]) #For each row (each time step), calculate the next stock price for each simulation  
            progress_bar(t / self.time_steps)                                                                   #We put the -1 in the Z index to ensure that we start from the 0'th row of the random matrix
        
        # TODO add a self.with_progress_bar attribute
        clear_progress_bar()

        self.price_paths = S


    def price_option(self):
        '''
        Abstract method for option pricing, has to be overridden by subclasses.
        '''
        raise NotImplementedError('Subclass must implement abstract method.')
