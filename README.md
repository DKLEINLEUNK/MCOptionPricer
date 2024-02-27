# MCOptionPricer
A comprehensive library focused on the financial applications for Monte Carlo methods.

## Table of Content
* [Instructions](#instructions)
* [Installation](#installation)
* [Project Structure](#structure)

## Instructions
```bash
usage: main.py [-h] [-S ASSET_PRICE] [-K STRIKE_PRICE] [-T MATURITY] [-r RATE] [-v VOLATILITY] [-n SIMULATIONS] [-t TIME_STEPS] [--plot]
               {call,put} {eu,us}

Monte Carlo Option Pricing Tool

positional arguments:
  {call,put}            Type of option (call or put)
  {eu,us}               Region of option (eu or us)

options:
  -h, --help            show this help message and exit
  -S ASSET_PRICE, --asset_price ASSET_PRICE
                        Initial stock price
  -K STRIKE_PRICE, --strike_price STRIKE_PRICE
                        Strike price
  -T MATURITY, --maturity MATURITY
                        Time to maturity (in years)
  -r RATE, --rate RATE  Risk-free interest rate
  -v VOLATILITY, --volatility VOLATILITY
                        Volatility
  -n SIMULATIONS, --simulations SIMULATIONS
                        Number of simulations
  -t TIME_STEPS, --time_steps TIME_STEPS
                        Number of time steps
  --plot                Plot option prices
```

## Installation
1. Clone the repository:
```bash
git clone https://github.com/DKLEINLEUNK/DES-simple-queues
```
2. Change directory to the cloned repository.
3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Structure
```
QuantMC/
├── LICENSE
├── main.py
├── MonteCarlo.py
├── Options.py
├── plotter.py
├── README.md
└── utils.py
```