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

from utils import phi
from MonteCarlo import MonteCarlo


class AsianOption(MonteCarlo):

    '''
    A class for pricing Asian options using Monte Carlo simulation.
    '''

    def price_option(self, method='arithmetic'):
        '''
        Description
        -----------
        Function to price an Asian option based on arithmetic averages using Monte Carlo simulation.
        
        Parameters
        ----------
        `method` : str
            `'arithmetic'` : Use the arithmetic average of the stock price (default).
            `'geometric'` : Use the geometric average of the stock price.
            `'control_variate'` : Use geometric as control variate in arithmetic.
        '''

        if method == 'arithmetic':
            return self.arithmetic_pricing()
        elif method == 'geometric':
            return self.geometric_pricing()
        elif method == 'control_variate':
            return self.control_variate_pricing()
        else:
            raise ValueError('Invalid method. Please choose arithmetic, geometric or control_variate.')
    

    def arithmetic_pricing(self):
        if self.price_paths is None:
            self.simulate_paths()

        S = self.price_paths  # collection of stock prices, shape: (simulations, time_steps)
        A = np.mean(S, axis=0)
        A = np.maximum(A - self.K, 0)
        C = np.exp(-self.r*self.T) * np.mean(A)
        return C


    def geometric_pricing(self):
        '''
        Analytical expression for the price of an Asian option based on geometric averages.
        '''
        alpha = 2*self.r / self.sigma**2
        adj_sigma = self.sigma * np.sqrt((2/self.T) * (1 - np.exp(-alpha * self.T)) / alpha)
        d1 = (np.log(self.S0/self.K) + (self.r + adj_sigma**2/2) * self.T) / (adj_sigma * np.sqrt(self.T))
        d2 = d1 - adj_sigma * np.sqrt(self.T)        
        C = np.exp(-self.r * self.T) * (self.S0 * np.exp((self.r - adj_sigma**2 / 2) * self.T) * phi(d1) - self.K * phi(d2))
        return C


    def control_variate_pricing(self):
        if self.price_paths is None:
            self.simulate_paths()
        
        S = self.price_paths  # collection of stock prices, shape: (simulations, time_steps)
        
        arithmetic_averages = np.mean(S, axis=0)
        geometric_averages = np.exp(np.mean(np.log(S), axis=0))
        expected_geometric = self.geometric_pricing()

        payoff_arithmetic = np.maximum(arithmetic_averages - self.K, 0)
        payoff_geometric = np.maximum(geometric_averages - self.K, 0)
        
        C_arithmetic = np.exp(-self.r * self.T) * np.mean(payoff_arithmetic)
        C_geometric = np.exp(-self.r * self.T) * np.mean(payoff_geometric)

        # find covariance and beta
        covariance = np.cov(payoff_arithmetic, payoff_geometric)
        beta = covariance[0, 1] / covariance[1, 1]
        C = C_arithmetic - beta * (C_geometric - expected_geometric)
        return C
    

    # def get_attributes(self):
    #     # Calculate or return pre-calculated attributes
    #     return {
    #         'price': [/* list of prices */],
    #         'delta': [/* list of deltas */],
    #         'gamma': [/* list of gammas */],
    #         'theta': [/* list of thetas */],
    #         'vega': [/* list of vegas */]
    #     }


def main():
    asian_option = AsianOption(
        S0=100,
        K=99,
        T=1,
        r=0.06,
        sigma=0.20,
        simulations=100_000,
        time_steps=252
    )

    # print(f'Price of Asian option based on arithmetic averages: {asian_option.price_option(method='arithmetic')}')
    # print(f'Price of Asian option based on geometric averages: {asian_option.price_option(method='geometric')}')
    print(f'Price of Asian option based on control variates: {asian_option.price_option(method="control_variate")}')
    

if __name__ == '__main__':
    main()
