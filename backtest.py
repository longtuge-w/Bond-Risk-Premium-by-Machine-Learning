import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from preprocess import required_maturities


# Class for backtesting direct bond trading strategy
class DirectBondTradeBacktester:
    def __init__(self, predicted_returns, dates, commission_rate, bond_prices, initial_capital=1000000, top_n=2):
        # Initialize backtester parameters
        self.predicted_returns = predicted_returns
        self.dates = dates
        self.commission_rate = commission_rate
        self.bond_prices = bond_prices
        self.initial_capital = initial_capital
        self.top_n = top_n
        self.positions = {}
        self.cash = [initial_capital]
        self.total_amount = [initial_capital]
        self.trades = []
        self.log = []
        self.returns = []
        self.flag = True

    # Method to execute trades based on predicted returns
    def execute_trade(self, idx):
        current_date = self.dates[idx]

        # Get predicted returns for the next month
        next_month_returns = {}
        for maturity in required_maturities[1:]:
            next_month_returns[maturity] = self.predicted_returns[maturity][self.model_name]['pred_return'][idx]

        # Sort maturities by predicted returns
        sorted_maturities = sorted(next_month_returns, key=next_month_returns.get, reverse=True)

        # Long top_n maturities and short bottom_n maturities
        long_positions = sorted_maturities[:self.top_n]
        short_positions = sorted_maturities[-self.top_n:]

        # Execute trades
        self.log.append(f"Long Positions: {long_positions}")
        self.log.append(f"Short Positions: {short_positions}")

        total_long_amount = 0
        total_short_amount = 0
        num_closed_positions = 0

        for maturity in required_maturities[1:]:
            current_price = self.bond_prices.loc[current_date, maturity]

            # Close positions that need to be closed
            if maturity in self.positions:
                if (maturity in long_positions and self.positions[maturity]['position'] == 'short') or \
                (maturity in short_positions and self.positions[maturity]['position'] == 'long') or \
                (maturity not in long_positions and maturity not in short_positions):
                    self.close_position(idx, maturity)
                    num_closed_positions += 1

        if num_closed_positions > 0 or self.flag:
            # Calculate the amount to allocate for each long and short position
            amount_per_position = self.cash[-1] / self.top_n

            # Open new long positions
            for maturity in long_positions:
                current_price = self.bond_prices.loc[current_date, maturity]
                if maturity not in self.positions or self.positions[maturity]['position'] == 'short':
                    self.open_position(idx, maturity, 'long', current_price, amount_per_position)
                    total_long_amount += amount_per_position

            # Open new short positions
            for maturity in short_positions:
                current_price = self.bond_prices.loc[current_date, maturity]
                if maturity not in self.positions or self.positions[maturity]['position'] == 'long':
                    self.open_position(idx, maturity, 'short', current_price, -amount_per_position)
                    total_short_amount += amount_per_position
        else:
            total_long_amount = self.last_long_amount
            total_short_amount = self.last_short_amount

        self.log.append(f"Total Long Amount: {total_long_amount:.2f}")
        self.log.append(f"Total Short Amount: {total_short_amount:.2f}")

        self.last_long_amount = total_long_amount
        self.last_short_amount = total_short_amount
        self.flag = False

        # Calculate total amount and return
        bond_value = sum(position['qty'] * self.bond_prices.loc[current_date, maturity] for maturity, position in self.positions.items())
        total_amount = self.cash[-1] + bond_value
        self.log.append(f"Total Amount: {total_amount:.2f}")
        self.total_amount.append(total_amount)

        if len(self.total_amount) > 1:
            current_return = (self.total_amount[-1] - self.total_amount[-2]) / self.total_amount[-2]
            self.returns.append(current_return)
            self.log.append(f"Return: {current_return:.4f}")
        else:
            self.returns.append(0)
            self.log.append("Return: 0.0000")

    # Method to open a new position
    def open_position(self, idx, maturity, position_type, current_price, amount):
        timestamp = self.dates[idx]
        qty = amount / current_price
        commission_fee = abs(qty) * current_price * self.commission_rate

        self.positions[maturity] = {
            'position': position_type,
            'qty': qty,
            'entry_price': current_price,
            'amount': amount
        }

        self.cash[-1] -= amount + commission_fee

        self.trades.append((f'open_{position_type}', idx, maturity, current_price, qty))
        self.log.append(f"Open {position_type.capitalize()} Position in {maturity} at Time {timestamp}")
        self.log.append(f"{'Bought' if position_type == 'long' else 'Sold'} {abs(qty):.2f} units at price {current_price:.4f}")
        self.log.append(f"Commission Fee: {commission_fee:.4f}")
        self.log.append(f"Current Cash: {self.cash[-1]:.2f}")
        self.log.append("---------------------------------------------------------------------")

    # Method to close an existing position
    def close_position(self, idx, maturity):
        timestamp = self.dates[idx]
        current_price = self.bond_prices.loc[timestamp, maturity]
        position_type = self.positions[maturity]['position']
        qty = self.positions[maturity]['qty']
        entry_price = self.positions[maturity]['entry_price']

        commission_fee = abs(qty) * current_price * self.commission_rate
        self.trades.append((f'close_{position_type}', idx, maturity, current_price, qty))

        if position_type == 'long':
            pnl = qty * (current_price - entry_price) - commission_fee
            self.cash[-1] += qty * current_price - commission_fee
        else:
            pnl = qty * (entry_price - current_price) - commission_fee
            self.cash[-1] -= abs(qty) * current_price + commission_fee

        del self.positions[maturity]

        self.log.append(f"Close {position_type.capitalize()} Position in {maturity} at Time {timestamp}")
        self.log.append(f"{'Sold' if position_type == 'long' else 'Bought'} {abs(qty):.2f} units at price {current_price:.4f}")
        self.log.append(f"PnL: {pnl:.2f}")
        self.log.append(f"Commission Fee: {commission_fee:.4f}")
        self.log.append(f"Current Cash: {self.cash[-1]:.2f}")
        self.log.append("---------------------------------------------------------------------")

    # Method to run the backtest for a specific model
    def run(self, model_name):
        self.model_name = model_name
        for idx in tqdm(range(len(self.dates) - 1), desc="Backtest Progress"):
            self.log.append("---------------------------------------------------------------------")
            self.log.append(f"Time {self.dates[idx]}:")
            self.log.append("---------------------------------------------------------------------")
            self.execute_trade(idx)

        self.returns = np.array(self.returns)

    # Method to calculate benchmark returns
    def calculate_benchmark_returns(self):
        benchmark_returns = []
        num_bonds = len(required_maturities[1:])
        amount_per_bond = {}

        for maturity in required_maturities[1:]:
            amount_per_bond[maturity] = self.initial_capital / num_bonds

        for idx in tqdm(range(len(self.dates) - 1), desc="Calculating Benchmark Returns"):
            current_date = self.dates[idx]
            next_date = self.dates[idx + 1]

            current_value = 0
            next_value = 0

            for maturity in required_maturities[1:]:
                current_price = self.bond_prices.loc[current_date, maturity]
                next_price = self.bond_prices.loc[next_date, maturity]

                current_value += amount_per_bond[maturity]
                next_value_i = amount_per_bond[maturity] / current_price * next_price
                next_value += next_value_i

                amount_per_bond[maturity] = next_value_i

            benchmark_return = (next_value - current_value) / current_value
            benchmark_returns.append(benchmark_return)

        benchmark_returns[0] -= self.commission_rate
        self.benchmark_returns = np.array(benchmark_returns)
        self.benchmark_cumulative_returns = np.cumprod(1 + self.benchmark_returns)

    # Method to display the log
    def display_log(self):
        for entry in self.log:
            self.log.append(entry)

    # Method to save the log to a file
    def save_log(self, filename):
        with open(filename, 'w') as f:
            for entry in self.log:
                f.write(entry + '\n')

    # Method to plot cumulative returns
    def plot_cumulative_returns(self, desc):
        self.cumulative_returns = np.cumprod(1 + self.returns)

        plt.figure(figsize=(10, 6))
        plt.plot(self.cumulative_returns, label='Strategy Returns')
        plt.plot(self.benchmark_cumulative_returns, label='Benchmark Returns')
        plt.xlabel('Time')
        plt.ylabel('Cumulative Returns')
        plt.title('Cumulative Returns')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'Graph/{desc} Cumulative Return.png')
        plt.show()

    # Method to calculate performance metrics
    def calculate_metrics(self):
        self.cumulative_return = self.cumulative_returns[-1] - 1
        self.sharpe_ratio = np.sqrt(12) * np.mean(self.returns) / np.std(self.returns)
        self.max_drawdown = np.max(1 - self.cumulative_returns / np.maximum.accumulate(self.cumulative_returns))
        self.win_ratio = np.sum(self.returns > 0) / np.sum(self.returns != 0)

    # Method to print performance metrics
    def print_metrics(self):
        print("----------------------------------------------------------------")
        print(f"Strategy info by model {self.model_name}")
        print(f"Cumulative Return: {self.cumulative_return:.2%}")
        print(f"Sharpe Ratio: {self.sharpe_ratio:.2f}")
        print(f"Max Drawdown: {self.max_drawdown:.2%}")
        print(f"Win Ratio: {self.win_ratio:.2%}")

        # Additional metrics for the strategy
        annual_return = np.mean(self.returns) * 12
        print(f"Annual Return: {annual_return:.2%}")

        volatility = np.std(self.returns) * np.sqrt(12)
        print(f"Annual Volatility: {volatility:.2%}")

        # Calculate and print metrics for the benchmark
        benchmark_cumulative_return = self.benchmark_cumulative_returns[-1] - 1
        benchmark_sharpe_ratio = np.sqrt(12) * np.mean(self.benchmark_returns) / np.std(self.benchmark_returns)
        benchmark_max_drawdown = np.max(1 - self.benchmark_cumulative_returns / np.maximum.accumulate(self.benchmark_cumulative_returns))
        benchmark_win_ratio = np.sum(self.benchmark_returns > 0) / np.sum(self.benchmark_returns != 0)

        print("----------------------------------------------------------------")
        print("Benchmark info")
        print(f"Cumulative Return: {benchmark_cumulative_return:.2%}")
        print(f"Sharpe Ratio: {benchmark_sharpe_ratio:.2f}")
        print(f"Max Drawdown: {benchmark_max_drawdown:.2%}")
        print(f"Win Ratio: {benchmark_win_ratio:.2%}")

        # Additional metrics for the benchmark
        benchmark_annual_return = np.mean(self.benchmark_returns) * 12
        print(f"Annual Return: {benchmark_annual_return:.2%}")

        benchmark_volatility = np.std(self.benchmark_returns) * np.sqrt(12)
        print(f"Annual Volatility: {benchmark_volatility:.2%}")