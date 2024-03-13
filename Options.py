import numpy as np
from MonteCarlo import MonteCarlo
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import lognorm


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
    

    def plot_stock_paths(self, save=False):
        '''
        Description
        -----------
        Visualises the first 50 paths generated in the Monte Carlo simulation. 
        '''
        num_rows = self.price_paths.shape[0]
        x_values = np.arange(num_rows)

        # plt.figure(figsize=(6, 5))
        fig = plt.figure(figsize=(7, 5))
        gs = GridSpec(1, 2, width_ratios=[5, 1], wspace=0.05)


        ax1 = plt.subplot(gs[0])
        for i in range(2_000):
            # plt.plot(x_values, self.price_paths[:, i])
            ax1.plot(x_values, self.price_paths[:, i], linewidth=1, alpha=0.25)
        ax1.set_xlabel("Time ($t$)", fontsize=14)
        ax1.set_ylabel("Stock Price ($S_t)$", fontsize=14)

        # final_prices = self.price_paths[-1, :]
        # plt.hist(final_prices, bins=30, orientation='horizontal', alpha=0.6, density=True)
        ax2 = plt.subplot(gs[1], sharey=ax1)
        final_prices = self.price_paths[-1, :]
        ax2.hist(final_prices, bins=30, orientation='horizontal', color='orange', alpha=0.7, density=True)
        
        plt.setp(ax2.get_xticklabels(), visible=False)
        plt.setp(ax2.get_yticklabels(), visible=False)

        plt.setp(ax1.get_yticklabels(), fontsize=12)
        plt.setp(ax1.get_xticklabels(), fontsize=12)
        
        plt.xticks(fontsize=12)
        plt.savefig("plots/plot_q1_stock_paths", dpi = 400)  if save else None
        plt.show()


    def compute_RMSE(self):

        '''
        Description
        -----------
        Computes the Route Mean Square Error for the estimated option price 
        '''

        payoff = np.maximum(self.K - self.price_paths[-1], 0) 

        discounted_payoffs = payoff * np.exp(-self.r * self.T) #Discount each payoff back to T0

        sd = np.std(discounted_payoffs, ddof= 1) #standard_deviation

        RMSE = sd/np.sqrt(self.simulations) #Route Mean Squared Error

        self.RMSE = RMSE

    
    def compute_CI(self, price_estimate):

        '''
        Description
        -----------
        Computes the confidence intervals for the estimated option price 
        '''
        
        Z_95 = 1.96 #Z score of 95% CI

        lower_95 = price_estimate - (Z_95 * self.RMSE) #Lower bound of 95% CI
        upper_95 = price_estimate + (Z_95 * self.RMSE) #Upper bound of 95% CI

        Z_99 = 2.576 #Z score of 95% CI

        lower_99 = price_estimate - (Z_99 * self.RMSE) #Lower bound of 95% CI
        upper_99 = price_estimate + (Z_99 * self.RMSE) #Upper bound of 95% CI


        return lower_95, upper_95, lower_99, upper_99



        





class EUCall(MonteCarlo):

    '''
    A class for pricing European call options using Monte Carlo simulation.
    '''

    def price_option(self, same_seed=False, state=None):
        if self.price_paths is None:
            self.simulate_paths(same_seed, state)
            
        
        payoff = np.maximum(self.price_paths[-1] - self.K, 0)
        price = np.exp(-self.r * self.T) * np.mean(payoff)
        return price
    

class EuUpAndOutCall(MonteCarlo):

    '''
    A class for pricing European up and out call options using Monte Carlo simulation.
    '''

    def price_option(self):

        if self.price_paths is None:
            self.simulate_paths_heston() #Use the heston model for the price path (implemented with Millstein scheme)
         
        cols_greater_than_barrier = np.any(self.price_paths >= self.barrier, axis = 0) #Checks row by row if any column contains a stock price >= B
        
        self.price_paths[-1,cols_greater_than_barrier] = 0 #Using boolean indexing. If any of the columns contained a S > B, the last element of that column will be set to 0 for the purpose of calculating the payoff of the barrier 
        
        payoff = np.maximum(self.price_paths[-1] - self.K, 0) #Subtract K from the last price of each simulation (price at time T)
        price = np.exp(-self.r * self.T) * np.mean(payoff)

        return price
    
    def compute_RMSE(self):

        '''
        Description
        -----------
        Computes the Route Mean Square Error for the estimated option price 
        '''

        cols_greater_than_barrier = np.any(self.price_paths >= self.barrier, axis = 0) #Checks row by row if any column contains a stock price >= B
        
        self.price_paths[-1,cols_greater_than_barrier] = 0 #Using boolean indexing. If any of the columns contained a S > B, the last element of that column will be set to 0 for the purpose of calculating the payoff of the barrier 
        
        payoff = np.maximum(self.price_paths[-1] - self.K, 0) #Subtract K from the last price of each simulation (price at time T)
        
        discounted_payoffs = payoff * np.exp(-self.r * self.T) #Discount each payoff back to T0

        sd = np.std(discounted_payoffs, ddof= 1) #standard_deviation

        RMSE = sd/np.sqrt(self.simulations) #Route Mean Squared Error

        self.RMSE = RMSE

    
    def compute_CI(self, price_estimate):

        '''
        Description
        -----------
        Computes the confidence intervals for the estimated option price 
        '''
        
        Z_95 = 1.96 #Z score of 95% CI

        lower_95 = price_estimate - (Z_95 * self.RMSE) #Lower bound of 95% CI
        upper_95 = price_estimate + (Z_95 * self.RMSE) #Upper bound of 95% CI

        Z_99 = 2.576 #Z score of 95% CI

        lower_99 = price_estimate - (Z_99 * self.RMSE) #Lower bound of 95% CI
        upper_99 = price_estimate + (Z_99 * self.RMSE) #Upper bound of 95% CI


        return lower_95, upper_95, lower_99, upper_99


    
class DigitalOption(MonteCarlo):
    
    '''
    A class for pricing
    '''
    
    def price_option(self, same_seed=False, state=None):
        if self.price_paths is None:
            self.simulate_paths(same_seed=same_seed, state=state)
        
        payoff = self.price_paths[-1] >= self.K
        price = np.exp(-self.r * self.T) * np.mean(payoff)

        return price


if __name__ == '__main__':

    # TESTING BARRIER LOGIC 
    # -----------------------------

    # mat = np.array([[1,2,1],
    #                [5,0,1],
    #                [1,2,3]])
    
    # print("BEFORE")
    # print(mat)
    
    # contains_greater_than_10_in_columns = np.any(mat > 4, axis=0)

    # mat[-1, contains_greater_than_10_in_columns] = 0

    # print("AFTER")
    # print(mat)

    # print(contains_greater_than_10_in_columns)

    #-----------------------------
    

    #print(testArr)

    ### Example for a put option ###
    # put = EUPut(
    #     S0=100,
    #     K=99,
    #     T=1,
    #     r=0.06,
    #     sigma=0.2,
    #     simulations=10_000,
    #     time_steps=250
    # )
    # put.price_option()
    # #print(f'Price for {put.name}: {put.price_option()}')
    
    
    # ### Example for a call option ###
    # call = EUCall(
    #     S0=100,
    #     K=99,
    #     T=1,
    #     r=0.06,
    #     sigma=0.2,
    #     simulations=10_000,
    #     time_steps=250
    # )
    # call.price_option()
    # #print(f'Price for {call.name}: {call.price_option()}')

    # upAndOutCall = EuUpAndOutCall(
    #     S0=100,
    #     K=100,
    #     T=1,
    #     r=0.06,
    #     sigma=0.2,
    #     simulations=10_000,
    #     time_steps=250,
    #     barrier = 120,
    #     theta = 0.04,
    #     kappa = 2, 
    #     epsilon = 0.1,
    #     rho = -0.7,
    #     V0 = 0.10

    # )

    # digital = DigitalOption(
    #     S0=100,
    #     K=99,
    #     T=1,
    #     r=0.06,
    #     sigma=0.2,
    #     simulations=10_000,
    #     time_steps=252,
    #     barrier = 60
    # )
    # print(f'Price of digital option: {digital.price_option()}')

    # upAndOutCall.price_option()
    # print(f'Price for {upAndOutCall.name}: {upAndOutCall.price_option()}')

    # S0, K, T, r, sigma,
    from plotter import test_option_pricing
    test_option_pricing(
        EUCall,
        strikes=np.linspace(50, 150, 20),  # Example strike prices range
        sigmas=np.linspace(0.1, 0.5, 20),  # Example volatilities range
        other_params={'S0': 100, 'T': 1, 'r': 0.06, 'simulations': 10_000, 'time_steps': 252}
    )
    pass