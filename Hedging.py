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

from utils import progress_bar, clear_progress_bar
from Options import EUPut, EUCall
from MonteCarlo import MonteCarlo


class Delta:

    def bump_and_revalue(self, option:MonteCarlo, n_simulation, bump_size=0.01, other_params=None):
        '''
        Estimate the hedge parameter in Monte Carlo using bump-and-revalue method.
        
        Parameters
        ----------
        option : MonteCarlo
            The option clas to be hedged.
        bump_size : float
            The size of the bump to the underlying price.
        model_params : dict
            The parameters for the underlying model.
        '''
        # original_prices = []
        # for _ in range(100):
        # Option.simulate_paths()
        # plt.plot(np.arange(Option.time_steps + 1), Option.price_paths[:,0])
        # plt.show()

        # Initialiate the option
        Option = option(simulations=n_simulation, **other_params)
        price = Option.price_option()

        # Bump the price and revalue the option
        Option.S0 = Option.S0 + bump_size
        print(f'Bumped S0 to {Option.S0}...')
        
        Option.price_paths = None  # reset price paths
        bumped_price = Option.price_option()
        
        print(f'Original price: {price}')
        print(f'Bumped price: {bumped_price}')

        delta = (bumped_price - price) / bump_size

        return delta
    

if __name__ == '__main__':

    option = EUCall

    model_params = {
        'S0': 100,
        'K' : 99,
        'T': 1,
        'r': 0.06,
        'sigma': 0.2,
        'time_steps': 252
    }
    
    import time

    start = time.time()

    delta = Delta().bump_and_revalue(
        option=option,
        n_simulation=1_000_000,
        bump_size=0.2,
        other_params=model_params
    )

    print(f'Time taken: {time.time() - start}')
    
    print(f'Delta for EUCall: {delta}')
