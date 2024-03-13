'''
File meant to contain the functions for hedging the portfolio. Assumes Black-Scholes model.

Delta Hedging European Options:
1. Estimate the hedge parameter in Monte Carlo using bump-and-revalue method.
    a. Use different seeds for bumped and unbumped estimates.
    b. Use the same seed for bumped and unbumped estimates.

Delta Hedging Digital Options:
1. Estimate the hedge parameter in Monte Carlo using pathwise method.
2. Estimate the hedge parameter in Monte Carlo using likelihood ratio method.
'''

import numpy as np
import matplotlib.pyplot as plt

from utils import progress_bar, clear_progress_bar, phi
from Options import EUPut, EUCall, DigitalOption
from MonteCarlo import MonteCarlo


def hedge_parameter_black_scholes(S0, K, T, r, sigma, **kwargs):
    '''
    Description
    -----------
    Calculates the delta hedge parameter for a European call option.
    
    Parameters
    ----------
    `S0` : float
        Initial stock price.
    `K` : float
        Strike price.
    `T` : float
        Maturity.
    `r` : float
        Risk-free interest rate.
    `sigma` : float
        Volatility.
    '''
    d1 = (np.log(S0/K) + (r + (sigma**2)/2)*T) / (T*np.sqrt(T))
    return phi(d1)


class Delta:


    def __init__(self, option:MonteCarlo, params=None):
        '''
        Parameters
        ----------
        option : MonteCarlo
            The option clas to be hedged.
        model_params : dict
            The parameters for the underlying model.
        '''
        self.Option = option(**params)
        self.params = params


    def bump_and_revalue(self, bump_size=0.2, same_seed=False):
        '''
        Estimate the hedge parameter in Monte Carlo using bump-and-revalue method.
        
        Parameters
        ----------
        bump_size : float
            The size of the bump to the underlying price.
        '''
        state = None
        if same_seed:
            np.random.seed(None)
            state = np.random.get_state()

        # Initialiate the option
        price = self.Option.price_option(same_seed=same_seed, state=state)

        # Bump the price and revalue the option
        self.Option.S0 = self.Option.S0 + bump_size
    
        self.Option.price_paths = None  # reset price paths
        bumped_price = self.Option.price_option(same_seed=same_seed, state=state)
 
        delta = (bumped_price - price) / bump_size

        # print(f'Delta: {delta}, Bumped price: {bumped_price}, Price: {price}')
        
        return delta
    

    def path_wise(self):

        epsilon = 1  # smoothing parameter TODO: make variable

        price = self.Option.price_option()

        S_T = self.Option.price_paths[-1]
        S_0 = self.Option.price_paths[0]
        K = self.Option.K
        T = self.Option.T
        r = self.Option.r

        # Apply smoothing to payoff

        # payoff = (np.exp(-(r+epsilon)*T) * (S_T - K) + np.exp(-(r-epsilon)*T) * (K - S_T)) / (2*epsilon)
        # payoff = np.exp(-(S_T - K)**2 / (2*epsilon**2)) / (np.sqrt(2*np.pi*epsilon**2))
        # payoff = np.exp(-(S_T - K)/epsilon) / (1 + np.exp(-(S_T - K)/epsilon))

        payoff = np.exp(-((S_T - K)/epsilon)) / (1 + np.exp(-((S_T - K)/epsilon)))**2  # <-- alex's derivation
        ratio = S_T /(epsilon*S_0)

        delta_path = np.exp(-r * T) * payoff * ratio  #A list of deltas for each stock path
        delta = np.mean(delta_path) #The final estimate for delta

        return delta


    def likelihood_ratio(self):
        '''
        Formulas
        --------
        Y = (log(S_T/S_0) - (r - sigma^2/2)T) / (S_0 * sigma^2 * sqrt(T))
        delta = E[e^(-rT) * 1{S_T>=K} * Y / (S_0 * sigma * sqrt(T))]
        '''
        price = self.Option.price_option()

        S_T = self.Option.price_paths[-1]
        S_0 = self.Option.price_paths[0]
        T = self.Option.T
        r = self.Option.r
        sigma = self.Option.sigma

        Y = (np.log(S_T/S_0) - (r - sigma**2/2)*T) / (S_0 * sigma**2 * T)  # note: this is a vector
        payoff = S_T >= self.Option.K
        delta = np.mean(np.exp(-r*T) * payoff * (Y)) # note: this was changed from the assignment
        
        return delta
    

    def likelihood_ratio_smooth(self):
        '''
        Formulas
        --------
        Y = (log(S_T/S_0) - (r - sigma^2/2)T) / (S_0 * sigma^2 * sqrt(T))
        delta = E[e^(-rT) * 1{S_T>=K} * Y / (S_0 * sigma * sqrt(T))]
        '''
        price = self.Option.price_option()

        S_T = self.Option.price_paths[-1]
        S_0 = self.Option.price_paths[0]
        K = self.Option.K
        T = self.Option.T
        r = self.Option.r
        sigma = self.Option.sigma

        Y = (np.log(S_T/S_0) - (r - sigma**2/2)*T) / (S_0 * sigma**2 * T)  # note: this is a vector
        payoff = np.exp(-((S_T - K)/1)) / (1 + np.exp(-((S_T - K)/1)))**2  # <-- alex's derivation
        ratio = S_T /(1*S_0)
        delta_path = np.exp(-r * T) * payoff * ratio  #A list of deltas for each stock path
        delta = np.mean(delta_path) # note: this was changed from the assignment
        
        return delta


    def export_deltas(self):
        pass
    

if __name__ == '__main__':

    # import time
    # start = time.time()
    
    # option = EUCall
    # model_params = {
    #     'S0': 100,
    #     'K' : 99,
    #     'T': 1,
    #     'r': 0.06,
    #     'sigma': 0.2,
    #     'simulations': 100_000,
    #     'time_steps': 252
    # }

    # option = DigitalOption
    # model_params = {
    #     'S0': 100,
    #     'K' : 99,
    #     'T': 1,
    #     'r': 0.06,
    #     'sigma': 0.2,
    #     'simulations': 100_000,
    #     'time_steps': 252
    # }

    # delta = Delta(
    #     option=option,
    #     params=model_params
    # )
    
    # path_wise_estimate = delta.path_wise()
    # print(f'Path wise estimate: {path_wise_estimate}')

    # likelihood_ratio_estimate = delta.likelihood_ratio()
    # print(f'Likelihood ratio estimate: {likelihood_ratio_estimate}')
    
    # delta_bump_estimates = []
    # delta_pathwise_estimates = []

    # for i in range(100):

    #     delta_bump = delta.bump_and_revalue()
    #     delta_bump_estimates.append(delta_bump)
    #     delta_path_wise = delta.path_wise()
    #     delta_pathwise_estimates.append(delta_path_wise)
    
    # plt.hist(delta_bump_estimates, bins = 30)
    # plt.show()

    
    

    
    # deltas = []

    # for _ in range(100):

    #     delta = Delta(
    #         option=option,
    #         params=model_params
    #     )
    
    #     hedge_param = delta.bump_and_revalue(bump_size=0.1, same_seed=True)
    #     deltas.append(hedge_param)

    # with open('deltas_same_seed_M_100_N_100_000.csv', 'w') as f:
    #     for delta in deltas:
    #         f.write(f'{delta}\n')
    

    # import pandas as pd
    
    # anal = hedge_parameter_black_scholes(**model_params)
    # data = pd.read_csv('deltas_same_seed_M_100_N_100_000.csv')

    # plt.hist(data, bins=20)
    # plt.axvline(anal, color='r', linestyle='--')
    # plt.show()

    # randomly initialize the RNG from some platform-dependent source of entropy
    np.random.seed(2)

    # get the initial state of the RNG
    st0 = np.random.get_state()
    # print(st0)

    # try 1: draw some random numbers
    print(np.random.randint(0, 100, 10))
    # [ 8 76 76 33 77 26  3  1 68 21]

    # set the state back to what it was originally
    # np.random.set_state(st0)

    # try 2:
    print(np.random.randint(0, 100, 10))

