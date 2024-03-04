'''
Module dedicated to pt 3 of the assignment: Variance Reduction

This module targets control variates using Asian options.

We assume the Black-Scholes model for the underlying asset price.

For valuation of an Asian option based on arithmetic averages we can use the value
of an Asian option based on geometric averages. This case can be solved analytically.

1.  Write a program for the price of an Asian option based on geometric averages using 
the analytical expression derived in A. Compare the values you obtain with the analytical
expression to those obtained by using Monte-Carlo simulations.

2.  To be done...
'''

# 1.a: Asian option pricing based on geometric & arithmetic averages
#-------------------------------------------------------------------
import numpy as np

from MonteCarlo import MonteCarlo


class AsianOption(MonteCarlo):

    '''
    A class for pricing Asian options using Monte Carlo simulation.
    '''

    def price_option(self):
        '''
        Description
        -----------
        Function to price an Asian option based on arithmetic averages using Monte Carlo simulation.
        '''
        if self.price_paths is None:
            self.simulate_paths()

        S = self.price_paths
        A = np.mean(S, axis=0)
        A = np.maximum(A - self.K, 0)
        C = np.exp(-self.r*self.T) * np.mean(A)
        return C


def geometric_pricing(S_0, r, T, K, sigma):
    '''
    Description
    -----------
    Function to price an Asian option based on geometric averages using the analytical expression.
    
    Parameters
    ----------
    S_0 : float
        The initial stock price.
    r : float
        The risk-free interest rate.
    T : float
        The time to maturity.
    K : float
        The strike price.
    sigma : float
        The volatility of the stock price.
    '''
    d1 = np.log(S_0/K) + (r + 0.5*sigma**2)*T / (sigma*np.sqrt(T))
    d2 = d1
    C = np.exp(-r*T) * (S_0*np.exp(r*T)*np.norm.cdf(d1) - K*np.norm.cdf(d2))
    return C
