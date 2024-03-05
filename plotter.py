'''
Assume each of your option pricing models has a method that returns a dictionary 
with keys representing different attributes (like price, delta, gamma, theta, 
and vega) and values being the numerical results.
'''
import matplotlib.pyplot as plt
import numpy as np

def plot_model_attributes(models):
    '''
    Plots attributes of option pricing models.

    Parameters:
    - models: a list of models or a single model object. Each model must have a `get_attributes` method 
      that returns a dictionary with keys like 'price', 'delta', 'gamma', 'theta', 'vega', and the corresponding values.

    This function does not return anything but shows a plot with the attributes of the provided models.
    '''
    if not isinstance(models, list):
        models = [models]  # Ensure models is a list for uniform processing
    
    attributes = ['price', 'delta', 'gamma', 'theta', 'vega']
    fig, axs = plt.subplots(len(attributes), 1, figsize=(10, 15))
    fig.tight_layout(pad=3.0)

    for i, attribute in enumerate(attributes):
        for model in models:
            attrs = model.get_attributes()
            axs[i].plot(attrs[attribute], label=model.__class__.__name__)
        axs[i].set_title(attribute.capitalize())
        axs[i].legend()

    plt.show()


def test_option_pricing(model_class, strikes, sigmas, other_params):
    """
    Plots option prices for varying strike prices and volatilities.
    
    Parameters:
    - model_class: The option pricing model class to test.
    - strikes: A list or array of strike prices.
    - sigmas: A list or array of volatilities.
    - other_params: A dictionary of other necessary parameters required by the model.
    """
    prices = np.zeros((len(strikes), len(sigmas)))
    
    for i, strike in enumerate(strikes):
        for j, sigma in enumerate(sigmas):
            model_instance = model_class(K=strike, sigma=sigma, **other_params)
            prices[i, j] = model_instance.price_option()

    Strike, Sigma = np.meshgrid(strikes, sigmas, indexing='ij')

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(Strike, Sigma, prices, cmap='viridis')

    ax.set_xlabel('Strike Price')
    ax.set_ylabel('Volatility')
    ax.set_zlabel('Option Price')
    ax.set_title('Option Prices for Varying Strike Prices and Volatilities')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    plt.show()