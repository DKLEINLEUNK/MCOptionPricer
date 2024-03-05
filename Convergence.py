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
    
    def value_option_black_scholes(S_t, K, tau, r, sigma):
        '''
        Description
        -----------
        Calculates the value of a European put option 
        using the Black-Scholes formula.

        Parameters
        ----------
        `S_t` : float
            Current stock price at t.
        `K` : float
            Strike price.
        `r` : float
            Risk-free interest rate.
        `vol` : float
            Volatility.
        `tau` : float
            Time to expiration, tau = T - t.
        '''
        d1 = (np.log(S_t/K) + (r + (sigma**2)/2)*tau) / (sigma*np.sqrt(tau))
        N_d1 = si.norm.cdf(d1)
        d2 = d1 - sigma*np.sqrt(tau)
        N_d2 = si.norm.cdf(d2)

        N_d1_negative = 1 - N_d1 #N(-d1)
        N_d2_negative = 1 - N_d2 #N(-d2)


        return K*np.exp(-r*tau)*N_d2_negative - S_t*N_d1_negative



    def convergence_to_black_scholes():
        
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

           lower_95, upper_95, lower_99, upper_99 = put.compute_CI() #Compute bounds for 95% and 99% CI
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
        
            
        plt.plot(num_of_trials, price_estimates, label = "MC Price", color = "blue")
        plt.plot(num_of_trials,lower_95_CI_values, linestyle='--', label = "Lower 95% CI", color = "black")
        plt.plot(num_of_trials,upper_95_CI_values, linestyle='--', label = "Upper 95% CI", color = "black")
        plt.plot(num_of_trials,lower_99_CI_values, linestyle='--', label = "Lower 99% CI", color = "grey")
        plt.plot(num_of_trials,upper_99_CI_values, linestyle='--', label = "Upper 99% CI", color = "grey")
        plt.axhline([bs_price], label="BS Price", color= "r")
        plt.xlabel("Number of trials")
        plt.xscale('log')
        plt.ylabel("Price estimates")
        plt.legend()
        plt.show()


if __name__ == "__main__":

   
    Convergence.convergence_to_black_scholes()

    # put = EUPut(
    #            S0=100,
    #             K=99,
    #             T=1,
    #             r=0.06,
    #             sigma=0.2,
    #             simulations= 100, #Vary number of trials
    #             time_steps=250
    #         )
    
    # price = put.price_option()
    
    # discounted_payoffs_sd = put.compute_payoff_standard_deviation()
    # print("CI bounds:  ", discounted_payoffs_sd)
    #print(discounted_payoffs)
    
    

