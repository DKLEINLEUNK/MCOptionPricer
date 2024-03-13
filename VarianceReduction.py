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
import matplotlib.pyplot as plt
import numpy as np

from utils import phi
from MonteCarlo import MonteCarlo


class AsianOption(MonteCarlo):

    '''
    A class for pricing Asian options using Monte Carlo simulation.
    '''

    def price_option(self, method='arithmetic', N=None):
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
            return self.geometric_pricing(N=N)
        elif method == 'analytical':
            return self.analytical_pricing()
        elif method == 'control_variate':
            return self.control_variate_pricing(N=N)
        else:
            raise ValueError('Invalid method. Please choose analytical, arithmetic, geometric or control_variate.')
    

    def arithmetic_pricing(self, as_array=False):
        if self.price_paths is None:
            self.simulate_paths()

        S = self.price_paths  # collection of stock prices, shape: (simulations, time_steps)
        S_T_bar = np.mean(S, axis=0)  # note: M = time_steps here

        H = np.maximum(S_T_bar - self.K, 0)
        
        pi = np.exp(-self.r*self.T) * H

        if as_array:
            return pi

        pi_se = (np.std(pi)) / np.sqrt(np.size(pi))

        return np.mean(pi), pi_se


    def geometric_pricing(self, N, as_array=False):
        if self.price_paths is None:
            self.simulate_paths()

        S = self.price_paths

        # select indices for the (iT/N) range
        indices = np.arange(0, self.time_steps, self.time_steps//N)
        S = S[indices]
        
        A_N_tilda = np.prod(S, axis=0)**(1/(N+1))
        
        # note for future: we only need to calculate stock prices of the (iT/N) range
        # print(A_N_tilda)

        H = np.maximum(A_N_tilda - self.K, 0)

        pi = np.exp(-self.r*self.T) * H

        if as_array:
            return pi

        pi_se = np.std(pi) / np.sqrt(np.size(pi))

        return np.mean(pi), pi_se


    def analytical_pricing(self):
        '''
        Analytical expression for the price of an Asian option based on geometric averages.
        '''
        N = self.time_steps  # assume number of observations = number of time steps

        sigma_tilda = self.sigma * np.sqrt((2*N + 1) / (6*(N + 1)))
        r_tilda = ((self.r - self.sigma**2/2) + (sigma_tilda**2)) / 2
        
        d1_tilda = (np.log(self.S0/self.K) + (r_tilda + sigma_tilda**2/2)*self.T) / np.sqrt(self.T*sigma_tilda)
        d2_tilda = (np.log(self.S0/self.K) - (r_tilda + sigma_tilda**2/2)*self.T) / np.sqrt(self.T*sigma_tilda)
        
        # C = np.exp((r_tilda-self.r)*self.T) * self.S0 * phi(d1_tilda) - self.K * phi(d2_tilda)

        pi = np.exp(-self.r*self.T) * self.S0 * np.exp(r_tilda*self.T) * phi(d1_tilda) - self.K * phi(d2_tilda)
        
        return pi


    def control_variate_pricing(self, N, as_array=False):
        if self.price_paths is None:
            self.simulate_paths()
        
        S = self.price_paths  # collection of stock prices, shape: (simulations, time_steps)

        pi_ari = self.arithmetic_pricing(as_array=True)
        pi_geo = self.geometric_pricing(N, as_array=True)
        pi_geo_exp = self.analytical_pricing()

        # C_arithmetic = np.exp(-self.r * self.T) * np.mean(payoff_arithmetic)
        # C_geometric = np.exp(-self.r * self.T) * np.mean(payoff_geometric)
        # print(C_arithmetic, C_geometric, expected_geometric)
        
        # find covariance and beta
        cov = np.cov(pi_ari, pi_geo)
        beta = -cov[0, 1] / cov[1, 1]

        # print(beta)

        pi_cv = pi_ari + beta * (pi_geo - pi_geo_exp)
        
        if as_array:
            return pi_cv
        
        pi_cv_se = np.std(pi_cv) / np.sqrt(np.size(pi_cv))
        
        return np.mean(pi_cv), pi_cv_se
    

if __name__ == '__main__':
    asian_option = AsianOption(
        S0=100,
        K=99,
        T=1,
        r=0.06,
        sigma=0.20,
        simulations=100_000,
        time_steps=252
    )

    # print(asian_option.price_option(method='analytical'))
    print(asian_option.price_option(method='geometric', N=25))
    print(asian_option.price_option(method='control_variate', N=25))
    # print(asian_option.price_option(method='arithmetic'))


