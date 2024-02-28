import matplotlib.pyplot as plt


# example of a function to plot option prices
def plot_option_prices(models, **kwargs):

    prices = [model.price_option(**kwargs) for model in models]
    model_names = [model.name for model in models]

    plt.figure(figsize=(10, 6))
    plt.bar(model_names, prices, color='skyblue')
    plt.xlabel('Option Pricing Model')
    plt.ylabel('Option Price')
    plt.title('Comparison of Option Pricing Models')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_stock_paths():
    pass