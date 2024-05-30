import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def read_csv(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            trades = line.strip().split(';')
            row = []
            for trade in trades:
                if trade:
                    profit, duration = map(float, trade.split(','))
                    row.append((profit, duration))
            data.append(row)
    return data


def calculate_metrics(trade_row):
    cumulative_profits = np.cumsum([trade[0] for trade in trade_row])
    max_drawdown = np.max(np.maximum.accumulate(cumulative_profits) - cumulative_profits)
    average_duration = np.mean([trade[1] for trade in trade_row]) if trade_row else 0
    max_duration = np.max([trade[1] for trade in trade_row]) if trade_row else 0
    total_profit = cumulative_profits[-1] if len(cumulative_profits) > 0 else 0
    recovery_factor = total_profit / max_drawdown if max_drawdown != 0 else np.inf
    return max_drawdown, average_duration, max_duration, recovery_factor, total_profit


def classify_strategy(total_profit, max_drawdown, average_duration, recovery_factor):
    if total_profit > 0 and recovery_factor > 1:
        return 1
    elif total_profit <= 0 and max_drawdown > -total_profit:
        return -1
    else:
        return 0


def plot_profits(data, indices):
    for idx in indices:
        trade_row = data[idx]
        cumulative_profits = np.cumsum([trade[0] for trade in trade_row])
        plt.plot(cumulative_profits, label=f'Series {idx + 1}')
    plt.xlabel('Trade Number')
    plt.ylabel('Cumulative Profit')
    plt.legend()
    plt.show()


def main():
    if len(sys.argv) != 3:
        print("Usage: python analyze_data.py <input_file_path> <output_file_path>")
        return

    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]
    data = read_csv(input_file_path)

    metrics = [calculate_metrics(row) for row in data]
    classifications = [classify_strategy(m[-1], m[0], m[1], m[3]) for m in metrics]

    df_metrics = pd.DataFrame(metrics, columns=['Max Drawdown', 'Average Duration', 'Max Duration', 'Recovery Factor',
                                                'Total Profit'])
    df_metrics['Classification'] = classifications

    df_metrics.to_csv(output_file_path, index=False)
    print(f"Results saved to {output_file_path}")

    sample_indices = np.random.choice(len(data), 3, replace=False)
    plot_profits(data, sample_indices)


if __name__ == "__main__":
    main()
