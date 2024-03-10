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

        C_se = np.sqrt(np.std(A) / np.sqrt(self.simulations))

        return C, C_se


    def geometric_pricing(self):
        '''
        Analytical expression for the price of an Asian option based on geometric averages.
        '''
        N = self.time_steps  # assume number of observations = number of time steps

        sigma_tilda = self.sigma * np.sqrt((2*N + 1) / (6*(N + 1)))
        r_tilda = ((self.r - self.sigma**2/2) + (sigma_tilda**2)) / 2
        d1_tilda = (np.log(self.S0/self.K) + (r_tilda + sigma_tilda**2/2)*self.T) / np.sqrt(self.T*sigma_tilda)
        d2_tilda = (np.log(self.S0/self.K) - (r_tilda + sigma_tilda**2/2)*self.T) / np.sqrt(self.T*sigma_tilda)
        # C = np.exp((r_tilda-self.r)*self.T) * self.S0 * phi(d1_tilda) - self.K * phi(d2_tilda)
        C = np.exp(-self.r*self.T) * self.S0 * np.exp(r_tilda*self.T) * phi(d1_tilda) - self.K * phi(d2_tilda)
        return C


    def control_variate_pricing(self):
        if self.price_paths is None:
            self.simulate_paths()
        
        S = self.price_paths  # collection of stock prices, shape: (simulations, time_steps)
        
        arithmetic_averages = np.mean(S, axis=0)
        geometric_averages = np.exp(np.mean(np.log(S), axis=0))
        expected_geometric = self.geometric_pricing()

        # print(geometric_averages)
        # print(arithmetic_averages)

        payoff_arithmetic = np.maximum(arithmetic_averages - self.K, 0)
        payoff_geometric = np.maximum(geometric_averages - self.K, 0)
        
        C_arithmetic = np.exp(-self.r * self.T) * np.mean(payoff_arithmetic)
        C_geometric = np.exp(-self.r * self.T) * np.mean(payoff_geometric)

        # find covariance and beta
        covariance = np.cov(payoff_arithmetic, payoff_geometric)
        beta = covariance[0, 1] / covariance[1, 1]
        C = C_arithmetic - beta * (C_geometric - expected_geometric)
        
        C_se = np.sqrt(np.std(C) / np.sqrt(self.simulations))
        
        return C, C_se
    

def plot_differences():
    
    arithmetic, geometric, both = [], [], []
    aritmetic_SE, geometric_SE, both_SE = [], [], []
    for K in range(90, 110, 1):
        asian_option = AsianOption(
            S0=100,
            K=K,
            T=1,
            r=0.06,
            sigma=0.20,
            simulations=10_000,
            time_steps=252
        )
        
        mu_ar, se_ar = asian_option.price_option(method='arithmetic')
        mu_ge = asian_option.price_option(method='geometric')
        mu_bo, se_bo = asian_option.price_option(method='control_variate')

        arithmetic.append(mu_ar)
        geometric.append(mu_ge)
        both.append(mu_bo)

        aritmetic_SE.append(se_ar)
        both_SE.append(se_bo)

    plt.figure(figsize=(5, 4))
    plt.style.use('seaborn-v0_8-bright')
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.plot(range(90, 110, 1), arithmetic, label='Arithmetic', linestyle='-', linewidth=2)
    plt.plot(range(90, 110, 1), geometric, label='Geometric', linestyle='-', linewidth=2)
    plt.plot(range(90, 110, 1), both, label='Control Variate', linestyle='-', linewidth=2)
    
    plt.fill_between(range(90, 110, 1), np.array(arithmetic) - np.array(aritmetic_SE) * 1.96, np.array(arithmetic) + np.array(aritmetic_SE) * 1.96, alpha=0.3)
    plt.fill_between(range(90, 110, 1), np.array(both) - np.array(both_SE) * 1.96, np.array(both) + np.array(both_SE) * 1.96, alpha=0.3)

    plt.xlabel('$K$', fontsize=14)
    plt.ylabel('Option Price', fontsize=14)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.show()

    # print(asian_option.price_option(method='arithmetic'))
    # print(asian_option.price_option(method='geometric'))
    # print(asian_option.price_option(method='control_variate'))


def main():
    
    plot_differences()

    # arithmetic, geometric, both = [], [], []
    # for _ in range(100):
    #     asian_option = AsianOption(
    #         S0=100,
    #         K=99,
    #         T=1,
    #         r=0.06,
    #         sigma=0.20,
    #         simulations=100_000,
    #         time_steps=252
    #     )
    #     arithmetic.append(asian_option.price_option(method='arithmetic'))
    #     geometric.append(asian_option.price_option(method='geometric'))
    #     both.append(asian_option.price_option(method='control_variate'))

    # print('Variances')
    # print('---------')
    # print(f'Arithmetic: {np.var(arithmetic)}')
    # print(f'Geometric: {np.var(geometric)}')
    # print(f'Control Variate: {np.var(both)}')

if __name__ == '__main__':
    main()
