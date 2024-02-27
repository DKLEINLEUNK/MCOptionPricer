from abc import ABC, abstractmethod

import numpy as np

from utils import progress_bar, clear_progress_bar


class MonteCarlo:

    '''
    A base class encapsulating core Monte Carlo simulation logic.
    '''

    def __init__(self, S0, K, T, r, sigma, simulations=10_000, time_steps=250):
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
        self.price_paths = None


    @property
    def name(self):
        return self.__class__.__name__


    def simulate_paths(self):
        '''
        Description
        -----------
        Generates paths for the underlying asset price (default using Euler scheme).
        '''
        dt = self.T / self.time_steps
        Z = np.random.standard_normal((self.time_steps, self.simulations))
        S = np.zeros((self.time_steps + 1, self.simulations))  # +1 for initial price S0
        S[0] = self.S0
        
        for t in range(1, self.time_steps + 1):
            S[t] = S[t - 1] * np.exp((self.r - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * Z[t - 1])
            progress_bar(t / self.time_steps)
        
        # TODO add a self.with_progress_bar attribute
        clear_progress_bar()

        self.price_paths = S


    def price_option(self):
        '''
        Abstract method for option pricing, has to be overridden by subclasses.
        '''
        raise NotImplementedError('Subclass must implement abstract method.')


class PathGenerator(ABC):
    @abstractmethod
    def simulate_paths(self, n_simulations, n_time_steps):
        pass


class BlackScholes(PathGenerator):

    def __init__(self, asset_price, strike_price, maturity, riskfree_rate, volatility) -> None:
        self.asset_price = asset_price
        self.strike_price = strike_price
        self.maturity = maturity
        self.riskfree_rate = riskfree_rate
        self.volatility = volatility
    
    def simulate_paths(self, n_simulations, n_time_steps):
        pass


class Heston(PathGenerator):

    def __init__(self, asset_price, barrier, strike_price, maturity, riskfree_rate, longterm_variance, reversion_rate, vol_of_vol, correlation) -> None:
        self.asset_price = asset_price
        self.barrier = barrier
        self.strike_price = strike_price
        self.maturity = maturity
        self.riskfree_rate = riskfree_rate
        self.longterm_variance = longterm_variance
        self.reversion_rate = reversion_rate
        self.vol_of_vol = vol_of_vol
        self.correlation = correlation
    
    def simulate_paths(self, n_simulations, n_time_steps):
        pass