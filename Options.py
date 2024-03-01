import numpy as np
from MonteCarlo import MonteCarlo


class EUPut(MonteCarlo):

    '''
    A class for pricing European put options using Monte Carlo simulation.
    '''

    def price_option(self):
        if self.price_paths is None:
            self.simulate_paths()
        
        payoff = np.maximum(self.K - self.price_paths[-1], 0)
        price = np.exp(-self.r * self.T) * np.mean(payoff)
        return price




class EUCall(MonteCarlo):

    '''
    A class for pricing European call options using Monte Carlo simulation.
    '''

    def price_option(self):
        if self.price_paths is None:
            self.simulate_paths()
            
        
        payoff = np.maximum(self.price_paths[-1] - self.K, 0)
        price = np.exp(-self.r * self.T) * np.mean(payoff)
        return price
    

class EuUpAndOutCall(MonteCarlo):

    '''
    A class for pricing European up and out call options using Monte Carlo simulation.
    '''

    def price_option(self):

        if self.price_paths is None:
            self.simulate_paths()
         
        cols_greater_than_barrier = np.any(self.price_paths >= self.barrier, axis = 0) #Checks row by row if any column contains a stock price >= B
        
        self.price_paths[-1,cols_greater_than_barrier] = 0 #Using boolean indexing. If any of the columns contained a S > B, the last element of that column will be set to 0 for the purpose of calculating the payoff of the barrier 
        
        payoff = np.maximum(self.price_paths[-1] - self.K, 0) #Subtract K from the last price of each simulation (price at time T)
        price = np.exp(-self.r * self.T) * np.mean(payoff)

        return price






if __name__ == '__main__':

    #TESTING BARRIER LOGIC 
    #-----------------------------

    mat = np.array([[1,2,1],
                   [5,0,1],
                   [1,2,3]])
    
    print("BEFORE")
    print(mat)
    
    contains_greater_than_10_in_columns = np.any(mat > 4, axis=0)

    mat[-1, contains_greater_than_10_in_columns] = 0

    print("AFTER")
    print(mat)

    print(contains_greater_than_10_in_columns)

    #-----------------------------
    

    #print(testArr)

    ### Example for a put option ###
    put = EUPut(
        S0=100,
        K=99,
        T=1,
        r=0.06,
        sigma=0.2,
        simulations=10_000,
        time_steps=250
    )
    put.price_option()
    #print(f'Price for {put.name}: {put.price_option()}')
    
    
    ### Example for a call option ###
    call = EUCall(
        S0=100,
        K=99,
        T=1,
        r=0.06,
        sigma=0.2,
        simulations=10_000,
        time_steps=250
    )
    call.price_option()
    print(f'Price for {call.name}: {call.price_option()}')

    upAndOutCall = EuUpAndOutCall(
        S0=100,
        K=99,
        T=1,
        r=0.06,
        sigma=0.2,
        simulations=10_000,
        time_steps=250,
        barrier= 140
    )

    upAndOutCall.price_option()
    print(f'Price for {upAndOutCall.name}: {upAndOutCall.price_option()}')

