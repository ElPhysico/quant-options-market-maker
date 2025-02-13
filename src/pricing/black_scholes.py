import numpy as np
from scipy.stats import norm


def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    """Compute the Black-Scholes price for European call or put options.

    Parameters:
        S (float): Current stock price
        K (float): Strike price
        T (float): Time to expiry (in years)
        r (float): Risk-free interest rate (annualized)
        sigma (float): Volatility of underlying asset (annualized)
        option_type (str): 'call' or 'put'

    Returns:
        float: Option price
    """
    if T < 0:
        raise ValueError('Time to expiry (T) cannot be negative')

    if T == 0:
        return max(0, S - K) if option_type == 'call' else max(0, K - S)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError('option_type must be "call" or "put"')