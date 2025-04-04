import math
import numpy as np


def optimal_trading_dp(prices, spread, volume_pct):
    n = len(prices)
    price_level_cnt = math.ceil(1 / volume_pct)
    left_over_pct = 1 - (price_level_cnt - 1) * volume_pct

    dp = [[float('-inf')] * (price_level_cnt * 2 + 1) for _ in range(n)]  # From -3 to 3, 7 positions
    action = [[''] * (price_level_cnt * 2 + 1) for _ in range(n)]  # To store actions

    # Initialize the starting position (no stock held)
    dp[0][price_level_cnt] = 0  # Start with no position, Cash is 0
    action[0][price_level_cnt] = ''  # No action at start

    def position(j):
        if j > price_level_cnt:
            position = min((j - price_level_cnt) * volume_pct, 1)
        elif j < price_level_cnt:
            position = max((j - price_level_cnt) * volume_pct, -1)
        else:
            position = 0
        return position

    def position_list(list):
        return np.array([position(x) for x in list])

    for i in range(1, n):
        for j in range(0, price_level_cnt * 2 + 1):
            # Calculate PnL for holding, buying, or selling
            hold = dp[i - 1][j] if dp[i - 1][j] != float('-inf') else float('-inf')
            if j == price_level_cnt * 2:
                buy = dp[i - 1][j - 1] - left_over_pct * prices[i - 1] - left_over_pct * spread if j > 0 else float(
                    '-inf')
            elif j == 1:
                buy = dp[i - 1][j - 1] - left_over_pct * prices[i - 1] - left_over_pct * spread if j > 0 else float(
                    '-inf')
            else:
                buy = dp[i - 1][j - 1] - volume_pct * prices[i - 1] - volume_pct * spread if j > 0 else float('-inf')

            if j == 0:
                sell = dp[i - 1][j + 1] + left_over_pct * prices[
                    i - 1] - left_over_pct * spread if j < price_level_cnt * 2 else float('-inf')
            elif j == price_level_cnt * 2 - 1:
                sell = dp[i - 1][j + 1] + left_over_pct * prices[
                    i - 1] - left_over_pct * spread if j < price_level_cnt * 2 else float('-inf')
            else:
                sell = dp[i - 1][j + 1] + volume_pct * prices[
                    i - 1] - volume_pct * spread if j < price_level_cnt * 2 else float('-inf')

            # Choose the action with the highest PnL

            hold_pnl = hold + (j - price_level_cnt) * position(j) * prices[i]
            buy_pnl = buy + (j - price_level_cnt) * position(j) * prices[i]
            sell_pnl = sell + (j - price_level_cnt) * position(j) * prices[i]

            # print(hold_pnl, buy_pnl, sell_pnl)
            best_action = max(hold_pnl, buy_pnl, sell_pnl)
            if best_action == hold_pnl:
                dp[i][j] = hold
            elif best_action == buy_pnl:
                dp[i][j] = buy
            else:
                dp[i][j] = sell

            if best_action == hold_pnl:
                action[i][j] = 'h'
            elif best_action == buy_pnl:
                action[i][j] = 'b'
            else:
                action[i][j] = 's'
    # Backtrack to find the sequence of actions
    trades_list = []
    # Start from the position with maximum PnL at time n-1

    pnl = np.array(dp[n - 1]) + (position_list(np.arange(0, price_level_cnt * 2 + 1)) * prices[n - 1])
    current_position = np.argmax(pnl)
    for i in range(n - 1, -1, -1):
        trades_list.append(action[i][current_position])
        if action[i][current_position] == 'b':
            current_position -= 1
        elif action[i][current_position] == 's':
            current_position += 1

    trades_list.reverse()
    trades_list.append('h')
    return dp, trades_list, pnl[np.argmax(pnl)]  # Return the actions and the maximum PnL


# Example usage
dp, trades, max_pnl = optimal_trading_dp(coconut_past_price, 0.99, 185 / 300)
# print(trades)
print("Max PnL:", max_pnl)