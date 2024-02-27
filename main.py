import argparse

import European as eu
from utils import cli_error_message


def main(type, region, S0, K, T, r, sigma, simulations, time_steps):
    # create the option based on user input
    if region == 'eu':  # TODO remove option checks, implement better handling
        if type == 'put':
            option = eu.Put(S0, K, T, r, sigma, simulations, time_steps)
        elif type == 'call':
            option = eu.Call(S0, K, T, r, sigma, simulations, time_steps)
    else:
        raise ValueError("Invalid region: Only European options are supported.")

    # price the option
    option_price = option.price_option()
    print(f"The European Call Option price is: {option_price}")


if __name__ == "__main__":
    # set up the command-line arguments
    parser = argparse.ArgumentParser(description="Monte Carlo Option Pricing Tool")
    parser.add_argument('type', type=str, choices=['call', 'put'], help='Type of option (call or put)')
    parser.add_argument('region', type=str, choices=['eu', 'us'], help='Region of option (eu or us)')
    parser.add_argument("-S", "--asset_price", type=float, default=100, help="Initial stock price")
    parser.add_argument("-K", "--strike_price", type=float, default=99, help="Strike price")
    parser.add_argument("-T", "--maturity", type=float, default=1, help="Time to maturity (in years)")
    parser.add_argument("-r", "--rate", type=float, default=0.06, help="Risk-free interest rate")
    parser.add_argument("-v", "--volatility", type=float, default=0.2, help="Volatility")
    parser.add_argument("-n", "--simulations", type=int, default=10**4, help="Number of simulations")
    parser.add_argument("-t", "--time_steps", type=int, default=252, help="Number of time steps")

    # read command-line arguments
    args = parser.parse_args()

    # print error if arguments invalid, else run main with provided arguments
    error = cli_error_message(args)
    if error:
        print(error)
    else:
        main(args.type, args.region, args.asset_price, args.strike_price, args.maturity, 
             args.rate, args.volatility, args.simulations, args.time_steps)
