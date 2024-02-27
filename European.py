import numpy as np
from MonteCarlo import MonteCarlo


class Put(MonteCarlo):

    '''
    A class for pricing European put options using Monte Carlo simulation.
    '''

    def price_option(self):
        if self.price_paths is None:
            self.euler_scheme()
        
        payoff = np.maximum(self.K - self.price_paths[-1], 0)
        price = np.exp(-self.r * self.T) * np.mean(payoff)
        return price


class Call(MonteCarlo):

    '''
    A class for pricing European call options using Monte Carlo simulation.
    '''

    def price_option(self):
        if self.price_paths is None:
            self.euler_scheme()
        
        payoff = np.maximum(self.price_paths[-1] - self.K, 0)
        price = np.exp(-self.r * self.T) * np.mean(payoff)
        return price


if __name__ == '__main__':

    ### Example for a put option ###
    put = Put(
        S0=100,
        K=99,
        T=1,
        r=0.06,
        sigma=0.2,
        simulations=10_000,
        time_steps=250
    )
    put.price_option()
    print(f'European Put Option Price: {put.price_option()}')
    
    
    ### Example for a call option ###
    call = Call(
        S0=100,
        K=99,
        T=1,
        r=0.06,
        sigma=0.2,
        simulations=10_000,
        time_steps=250
    )
    call.price_option()
    print(f'European Call Option Price: {call.price_option()}')