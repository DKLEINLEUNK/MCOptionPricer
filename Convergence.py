from Options import EUPut
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as si


class Convergence:

    """

    Description
    -----------
    A class containing all the methods to study the convergence of the Monte Carlo Method
    to the Black Scholes Solution.
    
    """

    #TODO : Perform multiple simulations to obtain the true estimate of the Put option 
    
    def analytical_solution(S0, K, T, r, sigma, **kwargs):
        '''
        Description
        -----------
        Calculates the value of a European put option using the Black-Scholes solution.

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
        d1 = (np.log(S0/K) + (r + (sigma**2)/2)*T) / (sigma*np.sqrt(T))
        N_d1 = si.norm.cdf(d1)
        d2 = d1 - sigma*np.sqrt(T)
        N_d2 = si.norm.cdf(d2)

        N_d1_negative = 1 - N_d1
        N_d2_negative = 1 - N_d2

        return K*np.exp(-r*T)*N_d2_negative - S0*N_d1_negative



    def convergence_to_black_scholes(save=False):
        
        '''
        Description
        -----------
        Compares estimates obtained from the MC method to the BS analytical solution
        for a put option.
        '''
        
        num_of_trials = [10, 100, 1000, 10_000, 100_000, 500_000]

        price_estimates = []
        lower_95_CI_values = []
        upper_95_CI_values = []
        lower_99_CI_values = []
        upper_99_CI_values = []

        
        
        for trials in num_of_trials:
           
           put = EUPut(
               S0=100,
                K=99,
                T=1,
                r=0.06,
                sigma=0.2,
                simulations= trials, #Vary number of trials
                time_steps=250
            )
           
           price = put.price_option() #Compute price estimate
           price_estimates.append(price)
           put.compute_RMSE() #Compute the RMSE
           
           lower_95, upper_95, lower_99, upper_99 = put.compute_CI(price) #Compute bounds for 95% and 99% CI
           lower_95_CI_values.append(lower_95)
           upper_95_CI_values.append(upper_95)
           lower_99_CI_values.append(lower_99)
           upper_99_CI_values.append(upper_99)
           

        #Compute BS Price
        bs_price = Convergence.value_option_black_scholes(
        S_t=100,
        K=99,
        tau=1,
        r=0.06,
        sigma=0.2
    )
        
        plt.figure(figsize=(5, 4))
            
        plt.plot(num_of_trials, price_estimates, label = "MC Price", color = "blue")
        plt.plot(num_of_trials,lower_95_CI_values, linestyle='--', label = "Lower 95% CI", color = "black")
        plt.plot(num_of_trials,upper_95_CI_values, linestyle='--', label = "Upper 95% CI", color = "black")
        plt.plot(num_of_trials,lower_99_CI_values, linestyle='--', label = "Lower 99% CI", color = "grey")
        plt.plot(num_of_trials,upper_99_CI_values, linestyle='--', label = "Upper 99% CI", color = "grey")
        plt.axhline([bs_price], label="BS Price", color= "r")
        plt.xlabel("Number of trials", fontsize=14)
        plt.xscale('log')
        plt.ylabel("Price estimates", fontsize=14)
        plt.yticks(fontsize=12)
        plt.xticks(fontsize=12)
        plt.legend(fontsize=12)
        plt.savefig("plots/plot_q1_convergence",bbox_inches='tight', dpi = 300)  if save else None
        plt.show()

    
    def standard_error():
        
        '''
        Description
        -----------
        Calculates the standard error of the MC estimate for increasing N
        '''

        num_of_trials = np.arange(1000,100_500,1000)
        RMSEs = []

        for trials in num_of_trials:

            put = EUPut(
               S0=100,
                K=99,
                T=1,
                r=0.06,
                sigma=0.2,
                simulations= trials, #Vary number of trials
                time_steps=250
            )
            put.price_option()
            put.compute_RMSE()
            RMSEs.append(put.RMSE)

        plt.figure(figsize=(5, 4))
        plt.plot(num_of_trials, RMSEs)
        plt.xlabel("Number of trials", fontsize = 14)
        plt.ylabel("Standard error", fontsize = 14)
        plt.yticks(fontsize=12)
        plt.xticks(fontsize=12)
        plt.legend(fontsize=12)
        plt.savefig("plots/plot_q1_standard_error",bbox_inches='tight', dpi = 300)
        plt.show()



    def strike_sensitivity():

        """

        Description
        -----------
        Determine sensitivity of the put option price to the strike price (K) 
        
        """

        Ks = np.arange(10,310,10)
        mc_prices = []
        bs_prices = []

        for K in Ks:

            put = EUPut(
            S0=100,
            K=K,
            T=1,
            r=0.06,
            sigma=0.2,
            simulations=10_000,
            time_steps=250,
        )
            
            #Compute MC price
            mc_price = put.price_option()
            mc_prices.append(mc_price)

            #Compute BS Price
            bs_price = Convergence.value_option_black_scholes(
            S_t=100,
            K=K,
            tau=1,
            r=0.06,
            sigma=0.2
        )   
            bs_prices.append(bs_price)

        price_differences = np.abs(np.array(mc_prices)-np.array(bs_prices)) 

        #MC price for different K's
        plt.figure(figsize=(5, 4))
        plt.plot(Ks, mc_prices)
        plt.xlabel("K",fontsize=14)
        plt.ylabel("Price", fontsize=14)
        plt.yticks(fontsize=12)
        plt.xticks(fontsize=12)
        plt.legend(fontsize=12)
        plt.savefig("plots/mc-strike-sensitivity",bbox_inches='tight', dpi = 300)
        plt.show()

        #Difference between MC price and BS price for different K's
        plt.figure(figsize=(5, 4))
        plt.plot(Ks, price_differences)
        plt.xlabel("K",fontsize=14)
        plt.ylabel('$|\\widehat{f}-f|$', fontsize=14)
        plt.yticks(fontsize=12)
        plt.xticks(fontsize=12)
        plt.legend(fontsize=12)
        plt.savefig("plots/bs-strike-sensitivity", bbox_inches='tight', dpi = 300)
        plt.show()




    def sigma_sensitivity():

        """

        Description
        -----------
        Determine sensitivity of the put option price to the volatility (sigma)
        """

        sigmas = np.arange(0,1.05,0.05)
        mc_prices = []
        bs_prices = []

        for sigma in sigmas:

            put = EUPut(
            S0=100,
            K=99,
            T=1,
            r=0.06,
            sigma=sigma,
            simulations=10_000,
            time_steps=250,
        )
            #Compute MC price
            mc_price = put.price_option()
            mc_prices.append(mc_price)
        

           #Compute BS Price
            bs_price = Convergence.value_option_black_scholes(
            S_t=100,
            K=99,
            tau=1,
            r=0.06,
            sigma=sigma
        )   
            bs_prices.append(bs_price)

        price_differences = np.abs(np.array(mc_prices)-np.array(bs_prices)) 

        #MC price for different sigmas
        plt.figure(figsize=(5, 4))
        plt.plot(sigmas, mc_prices)
        plt.xlabel("$\\sigma$",fontsize=14)
        plt.ylabel("Price", fontsize=14)
        plt.yticks(fontsize=12)
        plt.xticks(fontsize=12)
        plt.legend(fontsize=12)
        plt.savefig("plots/mc-sigma-sensitivity", bbox_inches='tight', dpi = 300)
        plt.show()

        #Difference between MC price and BS price for different sigmas
        plt.figure(figsize=(5, 4))
        plt.plot(sigmas, price_differences)
        plt.xlabel("$\\sigma$",fontsize=14)
        plt.ylabel('$|\\widehat{f}-f|$', fontsize=14)
        plt.yticks(fontsize=12)
        plt.xticks(fontsize=12)
        plt.legend(fontsize=12)
        plt.savefig("plots/bs-sigma-sensitivity",bbox_inches='tight', dpi = 300)
        plt.show()






if __name__ == "__main__":

    #TESTING implementation
    #-----------------------------
   
    #Convergence.convergence_to_black_scholes()
    Convergence.standard_error()
    #Convergence.strike_sensitivity()
    #Convergence.sigma_sensitivity()



    #-----------------------------
    

