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

    plt.figure(figsize=(5, 4))
    plt.style.use('seaborn-v0_8-bright')
    # plt.style.use('seaborn-v0_8-darkgrid')
    plt.imshow(prices, extent=[sigmas[0], sigmas[-1], strikes[0], strikes[-1]], origin='lower', aspect='auto', cmap='plasma')
    
    cb = plt.colorbar()
    cb.set_label(label='Option Price', fontsize=14)
    cb.ax.tick_params(labelsize=12)
    plt.xlabel('$\\sigma$', fontsize=14, fontweight='bold')
    plt.ylabel('$K$', fontsize=14, fontweight='bold')
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    # plt.legend(fontsize=12)
    # plt.show()
    plt.show()

