import numpy as np

from utils import progress_bar, clear_progress_bar

# TODO turn this into an abstract class

class MonteCarlo:

    '''
    A base class encapsulating core Monte Carlo simulation logic.
    '''

    def __init__(self, S0, K, T, r, sigma, simulations=10_000, time_steps=250, barrier = 0):
        '''
        Parameters
        ----------
        S0 : float
            Initial stock price
        K : float
            Strike price
        T : float
            Time to maturity (in years)
        r : float
            Risk-free interest rate
        sigma : float
            Volatility
        simulations : int
            Number of simulations
        time_steps : int
            Time steps
        '''
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.simulations = simulations
        self.time_steps = time_steps
        self.barrier = barrier
        self.price_paths = None
        


    @property
    def name(self):
        return self.__class__.__name__


    def simulate_paths(self):
        '''
        Description
        -----------
        Generates paths for the underlying asset price (default using Euler scheme).

        Parameters
        ----------
        with_progress_bar : bool
            If True, a progress bar will be displayed in the console.
        '''
        dt = self.T / self.time_steps
        Z = np.random.standard_normal((self.time_steps, self.simulations)) #A matrix consisiting of random numbers
        S = np.zeros((self.time_steps + 1, self.simulations))  # +1 for initial price S0
        S[0] = self.S0 #Initialises the entire first row of the S matrix to the starting stock price
        
        for t in range(1, self.time_steps + 1):
            S[t] = S[t - 1] * np.exp((self.r - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * Z[t - 1]) #For each row (each time step), calculate the next stock price for each simulation  
            progress_bar(t / self.time_steps)                                                                   #We put the -1 in the Z index to ensure that we start from the 0'th row of the random matrix
        
        # TODO add a self.with_progress_bar attribute
        clear_progress_bar()

        self.price_paths = S


    def simulate_paths_heston(self):

        """Generates paths for the underlying asset price according to the Heston model, implemented with the Millstein scheme"""
        
       
        theta = 0.04 #Long term variance
        kappa = 2 #Mean reversion rate
        epsilon = 0.1 #Vol-of-vol
        rho =  -0.7 #Correlation of Zv and Zs
        V0 = 0.10 #Initial volatility

        
        dt = self.T / self.time_steps

        #Generate correlated normally distributed variables with a correlation = rho
        Z1 = np.random.standard_normal((self.time_steps, self.simulations))
        Z2 = np.random.standard_normal((self.time_steps, self.simulations))

        Zv = Z1
        Zs = (rho*Z1) + np.sqrt(1-(rho**2))*Z2

        #Update variance 

        V = np.zeros((self.time_steps+1, self.simulations))   # +1 for initial variance V0
        S = np.zeros((self.time_steps+1, self.simulations))

        V[0] = V0 #Set first row to initial volatility
        S[0] = self.S0 #Set first row to initial stock prices

        for t in range(1, self.time_steps + 1): #Start looping from the next row after the first row
            
            #Update variance
            V[t] = V[t-1] + kappa*(theta - np.maximum(V[t-1],0))*dt + epsilon*np.sqrt(np.maximum(V[t-1],0)*dt)*Zv[t-1] + (1/4)*(epsilon**2)*dt*((Zv[t-1]**2) - 1)
            
            #Update stock price (USED GEOMETRIC MODEL SO THAT IT IS CONSISTENT WITH THE OTHER GENERATE STOCK PATHS METHOD)
            S[t] =  S[t-1] * np.exp((self.r - (1/2)*np.maximum(V[t],0))*dt + np.sqrt(np.maximum(V[t],0)*dt)*Zs[t-1])

            progress_bar(t / self.time_steps) 

            # TODO add a self.with_progress_bar attribute
        clear_progress_bar()
        
        self.price_paths = S



    def price_option(self):
        '''
        Description
        -----------

        Simulates stock paths of the Heston model using the Milstein scheme

        Parameters
        ----------
        with_progress_bar : bool
            If True, a progress bar will be displayed in the console.
        '''
        raise NotImplementedError('Subclass must implement abstract method.')


if __name__ == '__main__':

    #TESTING LOGIC
    #-----------------------------

    mat1 = np.array([[1,1,1],
                        [2,2,2]])
    
    mat2 = np.array([[10,20,10],
                        [10,20,10]])
    
    mat3 = (2* mat1) + mat2


    #print(mat1)
    #print(mat3)

    #-----------------------------

    #TESTING implementation
    #-----------------------------
    
    carlo = MonteCarlo(
        S0=100,
        K=99,
        T=1,
        r=0.06,
        sigma=0.2,
        simulations=5,
        time_steps=250,
     )
    
    carlo.simulate_paths_heston()

    print(carlo.price_paths)

    #-----------------------------