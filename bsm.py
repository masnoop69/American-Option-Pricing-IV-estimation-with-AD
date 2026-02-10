"""
Black-Scholes-Merton (BSM) Option Pricing and Greeks Library.

Functions for European option pricing, Greeks, and implied volatility.
"""
import time
import datetime
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from typing import Union
from fredapi import Fred

from scipy.stats import norm
from scipy.interpolate import griddata
import plotly.graph_objects as go
from plotly.subplots import make_subplots

NumericType = Union[float, np.ndarray]

# =============================================================================
# Fetch necessary Data
# =============================================================================


def fetch_fred_rf(key_path):
    with open(key_path, 'r') as file:
        api_key = file.read()
    fred = Fred(api_key = api_key)
    # lookback window cannnot be too wide because FRED data is not updated daily
    start_fred = (datetime.datetime.today() - datetime.timedelta(days = 30)).strftime('%Y-%m-%d')
    end_fred = datetime.datetime.today().strftime('%Y-%m-%d')
    # DCPF1M is the money market rate prevailiing in the US.
    money_market_rates = fred.get_series("DCPF1M", observation_start = start_fred, observation_end = end_fred)
    money_market_rates.dropna(inplace = True)
    rf = money_market_rates.iloc[-1] / 100 # find the latest value of the series
    print("Current Money Market Rates as of ", money_market_rates.index[-1].strftime('%Y-%m-%d'), "is: ", rf)
    return rf


def fetch_option_chains(ticker, T_start, T_end, type, price_range = 0.30, threshold_volume = 10):
    """
    Download all option chains with expiration up to T, number of years ahead, 
    for a given ticker and condense into a dataframe and filters out options 
    with low volume or NAN volume.

    We now include dividend yield extraction for calculation of expected yield

    Workflow:
    1. Get market price of option now (S0)
    2. Get all option chains with expiration up to T
    3. Filter out options with low volume or NAN volume
    """
    yticker = yf.Ticker(ticker)
    expirations = yticker.options
    try:
        S0 = yticker.fast_info['last_price']
        q = yticker.info.get('dividendYield') / 100 # forward dividend yield
    except:
        # fallback if fast_info fails, slower
        prices = yticker.history(period="1y")
        S0 = prices['Close'].iloc[-1]
        dividend_ttm = prices['Dividends'].sum()
        q = dividend_ttm / S0 # use ttm dividend yield as proxy for q

    option_chain_list = []
    for dates in range(len(expirations)):
        time_to_expiry = (datetime.datetime.strptime(expirations[dates], '%Y-%m-%d') - datetime.datetime.today()).days / 365
        if time_to_expiry < T_start: # skip options that have T < T_start
            continue
        if time_to_expiry > T_end: # break when T > T_end
            break
        if type == 'c':
            option_week_df = yticker.option_chain(str(expirations[dates])).calls
        elif type == 'p':
            option_week_df = yticker.option_chain(str(expirations[dates])).puts
        else:
            raise ValueError('Invalid option type, only accepts c or p')
        if option_week_df.empty:
            #print(f"No options found for {ticker} at {expirations[dates]}, skipping...")
            continue
        option_week_df['T'] = time_to_expiry
        option_chain_list.append(option_week_df[['strike', 'lastPrice', 'T', 'volume']])
        print(f"Obtained options for {ticker} at {expirations[dates]}")

        time.sleep(1) # prevent yfinance from blocking me
    option_chain = pd.concat(option_chain_list, ignore_index=True)

    # look at options centered around range about the current price
    option_chain_ranged = option_chain[(option_chain['strike'] / S0 - 1).abs() <= price_range]

    option_chain_ranged['S0'] = S0
    option_chain_ranged.columns = ['K', 'f', 'T', 'Volume','S0']
    option_chain_ranged['q'] = q / 100 # 

    # remove illiquid options, filter out options with below threshold volume
    # open interest data from yahoo is not reliable, hence we ignore it and use volume only
    option_chain_ranged.dropna(subset=['Volume'], inplace=True) # dropna for np.percentile to work
    option_chain_ranged = option_chain_ranged[(option_chain_ranged['Volume'] >= threshold_volume) & (option_chain_ranged['T'] > 0)]

    option_chain_ranged.reset_index(drop = True, inplace = True)

    return option_chain_ranged

# =============================================================================
# Core BSM Components
# =============================================================================

def bsm_d1(S0: NumericType, K: NumericType, T: NumericType, r: float, iv: NumericType) -> NumericType:
    """Calculate d1 term: [ln(S0/K) + (r + σ²/2)T] / (σ√T)"""
    return (np.log(S0 / K) + (r + iv ** 2 / 2) * T) / (iv * np.sqrt(T))


def bsm_d2(S0: NumericType, K: NumericType, T: NumericType, r: float, iv: NumericType) -> NumericType:
    """Calculate d2 term: d1 - σ√T"""
    return bsm_d1(S0, K, T, r, iv) - iv * np.sqrt(T)


# =============================================================================
# BSM Option Pricing
# =============================================================================

def bsm_price(
    S0: NumericType, K: NumericType, T: NumericType, 
    r: float, iv: NumericType, option_type: str = 'c'
) -> NumericType:
    """
    BSM European option price.
    
    Call: S0*N(d1) - K*e^(-rT)*N(d2)
    Put:  K*e^(-rT)*N(-d2) - S0*N(-d1)
    """
    d1 = bsm_d1(S0, K, T, r, iv)
    d2 = d1 - iv * np.sqrt(T)
    
    if option_type.lower() in ('c', 'call'):
        return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)


# =============================================================================
# BSM Greeks
# =============================================================================

def bsm_delta(
    S0: NumericType, K: NumericType, T: NumericType, 
    r: float, iv: NumericType, option_type: str = 'c'
) -> NumericType:
    """BSM Delta: ∂Price/∂S0. Call: N(d1), Put: N(d1) - 1"""
    d1 = bsm_d1(S0, K, T, r, iv)
    if option_type.lower() in ('c', 'call'):
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1


def bsm_gamma(
    S0: NumericType, K: NumericType, T: NumericType, 
    r: float, iv: NumericType
) -> NumericType:
    """BSM Gamma: ∂²Price/∂S0² = N'(d1) / (S0*σ*√T). Same for calls and puts."""
    d1 = bsm_d1(S0, K, T, r, iv)
    return norm.pdf(d1) / (S0 * iv * np.sqrt(T))


def bsm_vega(
    S0: NumericType, K: NumericType, T: NumericType, 
    r: float, iv: NumericType
) -> NumericType:
    """BSM Vega: ∂Price/∂σ = S0*N'(d1)*√T. Same for calls and puts."""
    d1 = bsm_d1(S0, K, T, r, iv)
    return S0 * norm.pdf(d1) * np.sqrt(T)


def bsm_theta(
    S0: NumericType, K: NumericType, T: NumericType, 
    r: float, iv: NumericType, option_type: str = 'c'
) -> NumericType:
    """BSM Theta (annualized): ∂Price/∂T. Divide by 365 for daily theta."""
    d1 = bsm_d1(S0, K, T, r, iv)
    d2 = d1 - iv * np.sqrt(T)
    first_term = -norm.pdf(d1) * S0 * iv / (2 * np.sqrt(T))
    
    if option_type.lower() in ('c', 'call'):
        return first_term - r * K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return first_term + r * K * np.exp(-r * T) * norm.cdf(-d2)


def bsm_rho(
    S0: NumericType, K: NumericType, T: NumericType, 
    r: float, iv: NumericType, option_type: str = 'c'
) -> NumericType:
    """BSM Rho: ∂Price/∂r. Call: K*T*e^(-rT)*N(d2), Put: -K*T*e^(-rT)*N(-d2)"""
    d2 = bsm_d2(S0, K, T, r, iv)
    
    if option_type.lower() in ('c', 'call'):
        return K * T * np.exp(-r * T) * norm.cdf(d2)
    else:
        return -K * T * np.exp(-r * T) * norm.cdf(-d2)


# =============================================================================
# BSM Implied Volatility
# =============================================================================

def bsm_iv(
    S0: float, K: float, T: float, r: float, 
    market_price: float, option_type: str = 'c',
    initial_guess: float = 0.5, max_iter: int = 200, tol: float = 1e-5
) -> float:
    """
    BSM Implied Volatility using Newton-Raphson with Brent fallback.
    
    Solves: BSM_Price(σ) = Market_Price for σ.
    Returns np.nan if no solution found.
    """
    is_call = option_type.lower() in ('c', 'call')
    intrinsic = max(0, S0 - K) if is_call else max(0, K - S0)
    
    if market_price < intrinsic:
        return np.nan
    
    iv_est = initial_guess
    for _ in range(max_iter):
        price = bsm_price(S0, K, T, r, iv_est, option_type)
        vega = bsm_vega(S0, K, T, r, iv_est)
        
        if abs(vega) < 1e-8:
            try:
                return brentq(lambda s: bsm_price(S0, K, T, r, s, option_type) - market_price, 0.0001, 3.0)
            except ValueError:
                return np.nan
        
        diff = price - market_price
        if abs(diff) < tol:
            return iv_est
        
        iv_est = max(0.0001, min(iv_est - diff / vega, 3.0))
    
    try:
        return brentq(lambda s: bsm_price(S0, K, T, r, s, option_type) - market_price, 0.0001, 3.0)
    except ValueError:
        return np.nan


def bsm_iv_vectorized(
    S0: np.ndarray, K: np.ndarray, T: np.ndarray, r: float, 
    market_prices: np.ndarray, option_type: str = 'c'
) -> np.ndarray:
    """Vectorized BSM IV estimation for arrays of options."""
    vec_func = np.vectorize(lambda s, k, t, p: bsm_iv(s, k, t, r, p, option_type))
    return vec_func(S0, K, T, market_prices)


# =============================================================================
# Vectorized DataFrame Operations
# =============================================================================

def bsm_get_greeks(
    df: pd.DataFrame, 
    rf: float, 
    option_type: str = 'c',
    inplace: bool = False
) -> pd.DataFrame:
    """
    Compute all BSM Greeks for a DataFrame of options.
    Compatible with American Binomial.ipynb structure.
    
    Expected columns: K, f (price), T, S0, q (dividend yield)
    Uses r = rf - q for expected return.
    
    Parameters:
        df: DataFrame with columns ['K', 'f', 'T', 'S0', 'q']
        rf: Risk-free rate (scalar)
        option_type: 'c' for calls, 'p' for puts
        inplace: If True, modify df in place; else return copy
    
    Returns:
        DataFrame with added columns: r, iv, delta, gamma, vega, theta, rho
    """
    if not inplace:
        df = df.copy()
    
    # Extract arrays (matching American Binomial.ipynb column names)
    S0 = df['S0'].values
    K = df['K'].values
    T = df['T'].values
    f = df['f'].values  # option price
    q = df['q'].values  # dividend yield
    
    # Calculate expected return (r = rf - q)
    r = rf - q
    df['r'] = r
    
    # Compute IV first using vectorized Newton-Raphson
    vec_iv = np.vectorize(lambda s, k, t, rate, price: bsm_iv(s, k, t, rate, price, option_type))
    iv = vec_iv(S0, K, T, r, f)
    df['iv'] = iv
    
    # Compute d1, d2
    d1 = (np.log(S0 / K) + (r + iv ** 2 / 2) * T) / (iv * np.sqrt(T))
    d2 = d1 - iv * np.sqrt(T)
    
    # Pre-compute common terms
    n_d1 = norm.cdf(d1)
    n_d2 = norm.cdf(d2)
    npdf_d1 = norm.pdf(d1)
    sqrt_T = np.sqrt(T)
    exp_rT = np.exp(-r * T)
    
    is_call = option_type.lower() in ('c', 'call')
    
    # Compute Greeks
    df['delta'] = n_d1 if is_call else n_d1 - 1
    df['gamma'] = npdf_d1 / (S0 * iv * sqrt_T)
    df['vega'] = S0 * npdf_d1 * sqrt_T
    
    if is_call:
        df['theta'] = -npdf_d1 * S0 * iv / (2 * sqrt_T) - r * K * exp_rT * n_d2
        df['rho'] = K * T * exp_rT * n_d2
    else:
        n_neg_d2 = norm.cdf(-d2)
        df['theta'] = -npdf_d1 * S0 * iv / (2 * sqrt_T) + r * K * exp_rT * n_neg_d2
        df['rho'] = -K * T * exp_rT * n_neg_d2
    
    # Drop rows with NaN IV (failed to compute)
    df.dropna(subset=['iv'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df = df[['K', 'f', 'T', 'S0', 'r', 'iv', 'delta', 'gamma', 'vega', 'theta', 'rho']]
    
    return df

# =============================================================================
# Visualize all greeks
# =============================================================================

def plot_greeks_surface(df, greek):
    if greek not in df.columns:
        raise ValueError(f"Column {greek} not found in dataframe")
        
    df['log_moneyness'] = np.log(df['S0'] / df['K'])

    x = df['log_moneyness'].values
    y = df['T'].values
    z = df[greek].values

    # Filter out outliers below the 2.5th percentile and above 97.5 percentile.
    mask = (np.abs(z) <= np.percentile(np.abs(z), 97.5))
    x, y, z = x[mask], y[mask], z[mask]

    # define grid of x = log_moneyness and y = T. high resolution not required.
    x_axis = np.linspace(x.min(), x.max(), 50)
    y_axis = np.linspace(y.min(), y.max(), 50)
    X_interpolated, Y_interpolated = np.meshgrid(x_axis, y_axis)

    # smooth interpolation with cubic method
    Z_interpolated = griddata((x, y), z, (X_interpolated, Y_interpolated), method='cubic')

    # plot surface
    fig = go.Figure(data=[go.Surface(
        z = Z_interpolated, 
        x = X_interpolated, 
        y = Y_interpolated,
        colorscale = 'Viridis',
        connectgaps = True
    )])

    fig.update_layout(
        title = f'{greek.upper()} Surface Plot with Interpolation',
        scene = dict(
            xaxis_title = 'Log Moneyness',
            yaxis_title = 'Time to Expiration (T)',
            zaxis_title = greek.upper()
        ),
        width=900,
        height=800
    )
    
    fig.show()

def plot_all_greeks_surface(df):
    """
    Plot all Greeks and IV surfaces in a 2x3 grid.
    
    Parameters:
        df: DataFrame with columns S0, K, T, iv, delta, gamma, vega, theta, rho
    """
    greeks = ['iv', 'delta', 'gamma', 'vega', 'theta', 'rho']
    
    df = df.copy()
    df['log_moneyness'] = np.log(df['S0'] / df['K'])
    
    # Create 2x3 subplot grid with 3D surface plots
    fig = make_subplots(
        rows=2, cols=3,
        specs=[[{'type': 'surface'}]*3, [{'type': 'surface'}]*3],
        subplot_titles=[g.upper() for g in greeks],
        horizontal_spacing=0.02,
        vertical_spacing=0.08
    )
    
    # Grid for interpolation
    x_axis = np.linspace(df['log_moneyness'].min(), df['log_moneyness'].max(), 50)
    y_axis = np.linspace(df['T'].min(), df['T'].max(), 50)
    X_interp, Y_interp = np.meshgrid(x_axis, y_axis)
    
    for idx, greek in enumerate(greeks):
        row = idx // 3 + 1
        col = idx % 3 + 1
        
        x = df['log_moneyness'].values
        y = df['T'].values
        z = df[greek].values
        
        # Filter outliers
        mask = np.abs(z) <= np.percentile(np.abs(z), 97.5)
        x_f, y_f, z_f = x[mask], y[mask], z[mask]
        
        # Interpolate
        Z_interp = griddata((x_f, y_f), z_f, (X_interp, Y_interp), method='cubic')
        
        fig.add_trace(
            go.Surface(z=Z_interp, x=X_interp, y=Y_interp, colorscale='Viridis', 
                       showscale=False, connectgaps=True),
            row=row, col=col
        )
    
    # Update layout
    fig.update_layout(
        title='Option Greeks Surface Plots',
        width=1400,
        height=900,
        margin=dict(l=10, r=10, t=60, b=10)
    )
    
    # Update all scene axes
    for i in range(1, 7):
        scene_name = f'scene{i}' if i > 1 else 'scene'
        fig.update_layout(**{
            scene_name: dict(
                xaxis_title='Log M',
                yaxis_title='T',
                zaxis_title='',
                xaxis=dict(tickfont=dict(size=8)),
                yaxis=dict(tickfont=dict(size=8)),
                zaxis=dict(tickfont=dict(size=8))
            )
        })
    
    fig.show()