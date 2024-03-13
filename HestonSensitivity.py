from Options import EuUpAndOutCall
from MonteCarlo import MonteCarlo
import numpy as np
import matplotlib.pyplot as plt


class sensitivityAnalysis:

    """

    Description
    -----------
    A class containing all the methods to determine the sensitivity of the 
    Eu up and out call option price to the relevant parameters of the Heston model
    
    """

    def barrier_sensitivity():

        """

        Description
        -----------
        Determine sensitivity of price to barrier level
        
        """

        barriers = np.arange(100,510,10)
        prices = []

        for barrier in barriers:

            upAndOutCall = EuUpAndOutCall(
            S0=100,
            K=100,
            T=1,
            r=0.05,
            sigma=0.2,
            simulations=10_000,
            time_steps=250,
            barrier = barrier, #Vary barrier
            theta = 0.04,
            kappa = 2, 
            epsilon = 0.1,
            rho =  -0.7,
            V0 = 0.20 
        )
            
            price = upAndOutCall.price_option()
            prices.append(price)
        
        plt.figure(figsize=(5, 4))
        plt.plot(barriers, prices, color = "red")
        plt.xlabel("$B$", fontsize=14)
        plt.ylabel("Price", fontsize=14)
        plt.yticks(fontsize=12)
        plt.xticks(fontsize=12)
        plt.legend(fontsize=12)
        plt.savefig("plots/heston-barrier-sensitivity", bbox_inches='tight', dpi = 300)
        plt.show()
        

    def correlation_sensitivity():

        """

        Description
        -----------
        Determine sensitivity of price to correlation (rho) 
        
        """

        rhos = np.arange(-1,1.10,0.05)
        prices = []

        for rho in rhos:

            upAndOutCall = EuUpAndOutCall(
            S0=100,
            K=100,
            T=1,
            r=0.05,
            sigma=0.2,
            simulations=10_000,
            time_steps=250,
            barrier = 120, 
            theta = 0.04,
            kappa = 2, 
            epsilon = 0.1,
            rho =  rho, #Vary rho's
            V0 = 0.20 
        )
            
            price = upAndOutCall.price_option()
            prices.append(price)
        
        plt.figure(figsize=(5, 4))
        plt.plot(rhos, prices, color = "red")
        plt.axvline(x=0, color='blue', linestyle=':')
        plt.xlabel("$\\rho$", fontsize = 14)
        plt.ylabel("Price", fontsize = 14)
        plt.yticks(fontsize=12)
        plt.xticks(fontsize=12)
        plt.legend(fontsize=12)
        plt.savefig("plots/heston-correlation-sensitivity",bbox_inches='tight',dpi = 300)
        plt.show()

        
    def vol_of_vol_sensitivity():

        """

        Description
        -----------
        Determine sensitivity of price to vol-of-vol (epsilon) 
        
        """

        epsilons = np.arange(0,1.10,0.05)
        prices = []

        for epsilon in epsilons:

            upAndOutCall = EuUpAndOutCall(
            S0=100,
            K=100,
            T=1,
            r=0.05,
            sigma=0.2,
            simulations=10_000,
            time_steps=250,
            barrier = 120, 
            theta = 0.04,
            kappa = 2, 
            epsilon = epsilon,
            rho =  -0.7, 
            V0 = 0.20 
        )
            
            price = upAndOutCall.price_option()
            prices.append(price)
        
        plt.figure(figsize=(5, 4))
        plt.plot(epsilons, prices, color = "red")        
        plt.xlabel("$\\epsilon$", fontsize = 14)
        plt.ylabel("Price", fontsize = 14)
        plt.yticks(fontsize=12)
        plt.xticks(fontsize=12)
        plt.legend(fontsize=12)
        plt.savefig("plots/heston-vol-of-vol-sensitivity",bbox_inches='tight', dpi = 300)
        plt.show()

        
        
        

            
            



if __name__ == "__main__":

    #Calling methods which display the plots for analysis
    sensitivityAnalysis.barrier_sensitivity()
    sensitivityAnalysis.correlation_sensitivity()
    sensitivityAnalysis.vol_of_vol_sensitivity()
    
        
        
    

