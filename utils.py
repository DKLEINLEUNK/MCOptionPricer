import sys


def cli_error_message(args):
    '''
    Checks if all command-line arguments are valid.
    '''
    if args.type not in ['call', 'put']:
        return "Invalid option type. Please choose 'call' or 'put'."
    if args.region not in ['eu', 'us']:
        return "Invalid option region. Please choose 'eu' or 'us'."
    if args.asset_price <= 0:
        return "Initial stock price must be positive."
    if args.strike_price <= 0:
        return "Strike price must be positive."
    if args.maturity <= 0:
        return "Time to maturity must be positive."
    if args.rate < 0:
        return "Risk-free interest rate must be non-negative."
    if args.volatility <= 0 or args.volatility > 1:
        return "Volatility must be between 0 and 1."
    if args.simulations <= 0:
        return "Number of simulations must be positive."
    if args.time_steps <= 0:
        return "Number of time steps must be positive."
    return None


def progress_bar(progress):
    bar_length = 50
    block = int(round(bar_length * progress))
    text = "\rProgress: [{0}] {1:.0f}%    ".format('█' * block + "-" * (bar_length - block), progress * 100)
    sys.stdout.write(text)
    sys.stdout.flush()

def clear_progress_bar():
    sys.stdout.write("\r")
    sys.stdout.flush()
    sys.stdout.write(" " * 100 + "\r")
    sys.stdout.flush()