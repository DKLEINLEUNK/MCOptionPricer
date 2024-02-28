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

    def bump_and_revalue(self, Option:MonteCarlo, bump_size=0.01):
        '''
        Estimate the hedge parameter in Monte Carlo using bump-and-revalue method.
        
        Parameters
        ----------
        Option : EUPut or EUCall
            The option object to hedge.
        bump_size : float
            The size of the bump in the underlying asset price.
        '''
        original_prices = []
        for _ in range(100):
            Option.simulate_paths()
            plt.plot(np.arange(Option.time_steps + 1), Option.price_paths[:,0])
            original_prices.append(Option.price_option())

        plt.show()

        original_price = np.mean(original_prices)  # find average over 100 estimates

        Option.S0 = Option.S0 + bump_size  # bump the price
        print(f'Bumped S0 to {Option.S0}...')
        
        bumped_prices = []
        for _ in range(100):
            Option.simulate_paths()
            bumped_prices.append(Option.price_option())
        
        bumped_price = np.mean(bumped_prices)  # find average over 100 estimates

        print(f'Original price: {original_price}')
        print(f'Bumped price: {bumped_price}')

        delta = (bumped_price - original_price) / bump_size

        return delta
    

if __name__ == '__main__':

    european_call = EUCall(
        S0=100,
        K=99,
        T=1,
        r=0.06,
        sigma=0.2,
        simulations=10_000,
        time_steps=252
    )
    delta = Delta().bump_and_revalue(european_call, bump_size=0.1)

    print(f'Delta for {european_call.name}: {delta}')
