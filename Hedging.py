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


    def bump_and_revalue(self, bump_size=0.01, same_seed=False):
        '''
        Estimate the hedge parameter in Monte Carlo using bump-and-revalue method.
        
        Parameters
        ----------
        bump_size : float
            The size of the bump to the underlying price.
        '''
        # TODO: look into using different seeds for bumped and unbumped estimates
        # TODO: use multiple simulations instead of just one
        # original_prices = []
        # for _ in range(100):
        # Option.simulate_paths()
        # plt.plot(np.arange(Option.time_steps + 1), Option.price_paths[:,0])
        # plt.show()

        if same_seed:
            seed = np.random.randint(100_000)
            np.random.seed(seed)

        # Initialiate the option
        price = self.Option.price_option()

        # Bump the price and revalue the option
        self.Option.S0 = self.Option.S0 + bump_size
        # print(f'Bumped S0 to {self.Option.S0}...')
        
        self.Option.price_paths = None  # reset price paths
        bumped_price = self.Option.price_option()
        
        # print(f'Original price: {price}')
        # print(f'Bumped price: {bumped_price}')

        delta = (bumped_price - price) / bump_size

        return delta
    
    def path_wise(self):

        price = self.Option.price_option()

        payoff = self.Option.price_paths[-1] >= self.Option.K

        ratio = self.Option.price_paths[-1]/self.Option.price_paths[0] #ST/S0

        delta_path = np.exp(-self.Option.r * self.Option.T) * payoff * ratio  #A list of deltas for each stock path

        delta = np.mean(delta_path) #The final estimate for delta

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

    option = DigitalOption
    model_params = {
        'S0': 100,
        'K' : 99,
        'T': 1,
        'r': 0.06,
        'sigma': 0.2,
        'simulations': 100_000,
        'time_steps': 252
    }



    delta = Delta(
        option=option,
        params=model_params
    )

    delta_bump_estimates = []
    delta_pathwise_estimates = []

    for i in range(100):

        delta_bump = delta.bump_and_revalue()
        delta_bump_estimates.append(delta_bump)
        delta_path_wise = delta.path_wise()
        delta_pathwise_estimates.append(delta_path_wise)
    
    plt.hist(delta_bump_estimates, bins = 30)
    plt.show()

    
    

    
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
    
    # print(f'Time taken: {time.time() - start}')
    
    # import pandas as pd
    
    # anal = hedge_parameter_black_scholes(**model_params)
    # data = pd.read_csv('deltas_same_seed_M_100_N_100_000.csv')

    # plt.hist(data, bins=20)
    # plt.axvline(anal, color='r', linestyle='--')
    # plt.show()
