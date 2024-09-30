import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as optimization
import random

# Constants
NUM_TRADING_DAYS = 252
NUM_PORTFOLIOS = 10000
STOCK_POOL_SIZE = 5  # Number of stocks in the final portfolio
PRICE_RANGE = (70, 100)
SHARPE_RATIO_RANGE = (0, 10)

# Fetch a random list of stock tickers from a broader index
def get_random_stock_list(pool_size, price_range):
    # Load tickers from a CSV file
    all_stocks = pd.read_csv('C:\Pcs\Python\QF/nasdaq_screener_1718837099670.csv')['Symbol'].tolist()
    all_stocks = [str(stock) for stock in all_stocks if '$' not in str(stock)]

    if pool_size > len(all_stocks):
        raise ValueError("Requested sample size is larger than the available stock list.")
    
    # Select random tickers
    all_stocks = random.sample(all_stocks, pool_size*100)  # Fetch more to account for price filtering


    
    # Filter stocks based on price range
    filtered_stocks = []
    for stock in all_stocks:
        ticker = yf.Ticker(stock)
        # print(ticker)
        try:
            price_data = ticker.history(period="1d")
            if not price_data.empty:
                price = price_data['Close'].iloc[0]
                if price_range[0] <= price <= price_range[1]:
                    filtered_stocks.append(stock)
                if len(filtered_stocks) >= pool_size:
                    break
        except Exception as e:
            print(f"Could not retrieve data for {stock}: {e}")
    
    if len(filtered_stocks) < pool_size:
        raise ValueError("Not enough stocks in the specified price range.")
    
    return filtered_stocks

# Historical data fetching
def download_data(stocks, start_date, end_date):
    stock_data = {}

    for stock in stocks:
        ticker = yf.Ticker(stock)
        stock_data[stock] = ticker.history(start=start_date, end=end_date)['Close']

    return pd.DataFrame(stock_data)

# Calculate log returns
def calculate_return(data):
    log_return = np.log(data / data.shift(1))
    return log_return[1:]

# Generate random portfolios
def generate_portfolios(returns):
    portfolio_means = []
    portfolio_risks = []
    portfolio_weights = []
    portfolio_sharpe_ratios = []

    for _ in range(NUM_PORTFOLIOS):
        weights = np.random.random(len(stocks))
        weights /= np.sum(weights)
        portfolio_weights.append(weights)
        portfolio_means.append(np.sum(returns.mean() * weights) * NUM_TRADING_DAYS)
        portfolio_risks.append(np.sqrt(np.dot(weights.T, np.dot(returns.cov() * NUM_TRADING_DAYS, weights))))
        portfolio_sharpe_ratios.append(statistics(weights, returns)[2])

    return np.array(portfolio_weights), np.array(portfolio_means), np.array(portfolio_risks), np.array(portfolio_sharpe_ratios)

# Portfolio statistics
def statistics(weights, returns):
    portfolio_return = np.sum(returns.mean() * weights) * NUM_TRADING_DAYS
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * NUM_TRADING_DAYS, weights)))
    sharpe_ratio = portfolio_return / portfolio_volatility
    return np.array([portfolio_return, portfolio_volatility, sharpe_ratio])

# Optimize portfolio
def min_function_sharpe(weights, returns):
    return -statistics(weights, returns)[2]

def optimize_portfolio(weights, returns):
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(len(stocks)))
    return optimization.minimize(fun=min_function_sharpe, x0=weights[0], args=returns, method='SLSQP', bounds=bounds, constraints=constraints)

# Filter portfolios by Sharpe ratio
def filter_portfolios_by_sharpe_ratio(weights, returns, means, risks, sharpe_ratios, sharpe_range):
    filtered_weights = []
    filtered_means = []
    filtered_risks = []
    filtered_sharpe_ratios = []

    for i in range(len(sharpe_ratios)):
        if sharpe_range[0] <= sharpe_ratios[i] <= sharpe_range[1]:
            filtered_weights.append(weights[i])
            filtered_means.append(means[i])
            filtered_risks.append(risks[i])
            filtered_sharpe_ratios.append(sharpe_ratios[i])

    return np.array(filtered_weights), np.array(filtered_means), np.array(filtered_risks), np.array(filtered_sharpe_ratios)

# Print the optimal portfolio
def print_optimal_portfolio(optimum, returns):
    print("Optimal portfolio weights: ", optimum['x'].round(3))
    print("Expected return, volatility, and Sharpe ratio: ", statistics(optimum['x'].round(3), returns))

# Show the optimal portfolio
def show_optimal_portfolio(opt, rets, portfolio_rets, portfolio_vols):
    plt.figure(figsize=(10, 6))
    plt.scatter(portfolio_vols, portfolio_rets, c=portfolio_rets / portfolio_vols, marker='o')
    plt.grid(True)
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.colorbar(label='Sharpe Ratio')
    plt.plot(statistics(opt['x'], rets)[1], statistics(opt['x'], rets)[0], 'g*', markersize=20.0)
    plt.title('Optimal Portfolio Based on Sharpe Ratio')
    plt.show()

# Main script execution
if __name__ == '__main__':
    # Get stock list and filter by price range
    filtered_stocks = get_random_stock_list(STOCK_POOL_SIZE, PRICE_RANGE)
    
    # Sample random stocks from the filtered list
    stocks = filtered_stocks
    print(f"Selected stocks: {stocks}")

    # Download historical data
    start_date = '2023-06-24'
    end_date = '2024-06-24'
    dataset = download_data(stocks, start_date, end_date)
    
    # Calculate returns
    log_daily_returns = calculate_return(dataset)
    
    # Generate portfolios
    pweights, means, risks, sharpe_ratios = generate_portfolios(log_daily_returns)
    
    # Filter portfolios based on Sharpe ratio
    filtered_weights, filtered_means, filtered_risks, filtered_sharpe_ratios = filter_portfolios_by_sharpe_ratio(
        pweights, log_daily_returns, means, risks, sharpe_ratios, SHARPE_RATIO_RANGE)
    
    # Optimize the filtered portfolio
    if len(filtered_weights) > 0:
        optimum = optimize_portfolio(filtered_weights, log_daily_returns)
        print_optimal_portfolio(optimum, log_daily_returns)
        show_optimal_portfolio(optimum, log_daily_returns, filtered_means, filtered_risks)
    else:
        print("No portfolios found within the specified Sharpe ratio range.")