# options trading v1

# import math
# import numpy as np
# from datamodel import OrderDepth, UserId, TradingState, Order
# from typing import List
# import string

# class Trader:
#     # Constants
#     VOUCHERS = [
#         "VOLCANIC_ROCK_VOUCHER_9500",
#         "VOLCANIC_ROCK_VOUCHER_9750",
#         "VOLCANIC_ROCK_VOUCHER_10000",
#         "VOLCANIC_ROCK_VOUCHER_10250",
#         "VOLCANIC_ROCK_VOUCHER_10500"
#     ]
#     STRIKES = {voucher: int(voucher.split('_')[-1]) for voucher in VOUCHERS}
#     POSITION_LIMIT = 200
#     UNDERLYING = "VOLCANIC_ROCK"
#     DAYS_PER_YEAR = 252
#     THRESHOLD = 0.01
#     TRADE_QUANTITY = 10

#     @staticmethod
#     def norm_cdf(x):
#         return 0.5 * (1 + math.erf(x / math.sqrt(2)))

#     @staticmethod
#     def norm_pdf(x):
#         return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

#     @staticmethod
#     def bs_call(S, K, T, sigma, r=0):
#         if T <= 0:
#             return max(S - K, 0)
#         d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
#         d2 = d1 - sigma * math.sqrt(T)
#         return S * Trader.norm_cdf(d1) - K * math.exp(-r * T) * Trader.norm_cdf(d2)

#     @staticmethod
#     def bs_vega(S, K, T, sigma, r=0):
#         if T <= 0:
#             return 0
#         d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
#         return S * math.sqrt(T) * Trader.norm_pdf(d1)

#     @staticmethod
#     def implied_vol(S, K, T, C_market, r=0, tol=1e-6, max_iter=100):
#         if C_market <= 0 or S <= 0 or K <= 0 or T <= 0:
#             return None
#         sigma = 0.5
#         for _ in range(max_iter):
#             C_bs = Trader.bs_call(S, K, T, sigma, r)
#             vega = Trader.bs_vega(S, K, T, sigma, r)
#             if vega == 0:
#                 return None
#             diff = C_bs - C_market
#             if abs(diff) < tol:
#                 return sigma
#             sigma -= diff / vega
#             if sigma <= 0:
#                 sigma = 0.0001
#         return None

#     @staticmethod
#     def get_mid_price(order_depth):
#         if order_depth.buy_orders and order_depth.sell_orders:
#             best_bid = max(order_depth.buy_orders.keys())
#             best_ask = min(order_depth.sell_orders.keys())
#             return (best_bid + best_ask) / 2
#         return None

#     def run(self, state: TradingState):
#         print("traderData: " + state.traderData)
#         print("Observations: " + str(state.observations))

#         result = {}
#         day = state.timestamp  # Assuming timestamp starts at 0 for day 1
#         TTE_days = 7 - day
#         if TTE_days <= 0:
#             traderData = "EXPIRED"
#             return result, 0, traderData
#         T = TTE_days / self.DAYS_PER_YEAR

#         # Get underlying price
#         underlying_depth = state.order_depths.get(self.UNDERLYING, OrderDepth())
#         St = self.get_mid_price(underlying_depth)
#         if St is None:
#             traderData = "NO_UNDERLYING_PRICE"
#             return result, 0, traderData

#         # Collect data for curve fitting
#         m_t_list = []
#         v_t_list = []
#         for voucher in self.VOUCHERS:
#             if voucher not in state.order_depths:
#                 continue
#             order_depth = state.order_depths[voucher]
#             Vt = self.get_mid_price(order_depth)
#             if Vt is None:
#                 continue
#             K = self.STRIKES[voucher]
#             try:
#                 m_t = math.log(K / St) / math.sqrt(T)
#                 v_t = self.implied_vol(St, K, T, Vt)
#                 if v_t is not None:
#                     m_t_list.append(m_t)
#                     v_t_list.append(v_t)
#             except (ValueError, ZeroDivisionError):
#                 continue

#         if len(m_t_list) < 3:
#             traderData = "INSUFFICIENT_DATA"
#             return result, 0, traderData

#         # Fit parabolic curve
#         coef = np.polyfit(m_t_list, v_t_list, 2)

#         # Process each product in order_depths
#         for product in state.order_depths:
#             if product not in self.VOUCHERS:
#                 continue  # Only trade vouchers
#             order_depth = state.order_depths[product]
#             orders = []
#             Vt = self.get_mid_price(order_depth)
#             if Vt is None:
#                 continue
#             K = self.STRIKES[product]
#             try:
#                 m_t = math.log(K / St) / math.sqrt(T)
#                 v_t = self.implied_vol(St, K, T, Vt)
#                 if v_t is None:
#                     continue
#                 fitted_v_t = np.polyval(coef, m_t)
#                 deviation = v_t - fitted_v_t
#                 position = state.position.get(product, 0)

#                 # Calculate acceptable price (mid-price for logging)
#                 acceptable_price = Vt
#                 print(f"Acceptable price : {acceptable_price}")
#                 print(f"Buy Order depth : {len(order_depth.buy_orders)}, Sell order depth : {len(order_depth.sell_orders)}")

#                 if deviation > self.THRESHOLD and position > -self.POSITION_LIMIT:
#                     if order_depth.buy_orders:
#                         best_bid = max(order_depth.buy_orders.keys())
#                         best_bid_amount = order_depth.buy_orders[best_bid]
#                         max_sell_qty = position + self.POSITION_LIMIT
#                         qty = min(self.TRADE_QUANTITY, best_bid_amount, max_sell_qty)
#                         if qty > 0:
#                             print(f"SELL {qty}x {best_bid}")
#                             orders.append(Order(product, best_bid, -qty))

#                 elif deviation < -self.THRESHOLD and position < self.POSITION_LIMIT:
#                     if order_depth.sell_orders:
#                         best_ask = min(order_depth.sell_orders.keys())
#                         best_ask_amount = order_depth.sell_orders[best_ask]
#                         max_buy_qty = self.POSITION_LIMIT - position
#                         qty = min(self.TRADE_QUANTITY, best_ask_amount, max_buy_qty)
#                         if qty > 0:
#                             print(f"BUY {qty}x {best_ask}")
#                             orders.append(Order(product, best_ask, qty))

#                 if orders:
#                     result[product] = orders
#             except (ValueError, ZeroDivisionError):
#                 continue

#         traderData = "SAMPLE"  # Keeping as per template
#         conversions = 0  # No conversions needed
#         return result, conversions, traderData

# options trading v2

# import json
# import math
# from datamodel import OrderDepth, UserId, TradingState, Order
# from typing import List

# def norm_pdf(x):
#     return (1.0 / math.sqrt(2 * math.pi)) * math.exp(-x**2 / 2)

# def norm_cdf(x):
#     a1 = 0.254829592
#     a2 = -0.284496736
#     a3 = 1.421413741
#     a4 = -1.453152027
#     a5 = 1.061405429
#     p = 0.3275911

#     sign = 1
#     if x < 0:
#         sign = -1
#     x = abs(x) / math.sqrt(2.0)
#     t = 1.0 / (1.0 + p * x)
#     y = 1.0 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t) * math.exp(-x*x)
#     return 0.5 * (1.0 + sign * y)

# def black_scholes_call(S, K, T, sigma, r=0):
#     if T <= 0:
#         return max(S - K, 0)
#     sqrt_T = math.sqrt(T)
#     d1 = (math.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * sqrt_T)
#     d2 = d1 - sigma * sqrt_T
#     call_price = S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
#     return call_price

# def vega(S, K, T, sigma, r=0):
#     if T <= 0:
#         return 0
#     sqrt_T = math.sqrt(T)
#     d1 = (math.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * sqrt_T)
#     return S * norm_pdf(d1) * sqrt_T

# def compute_implied_vol(S, K, T, market_price, r=0, max_iter=100, tol=1e-6):
#     intrinsic = max(S - K, 0.0)
#     if market_price < intrinsic - 1e-8 or market_price > S + 1e-8 or T <= 0:
#         return None
#     sigma = 0.5
#     for _ in range(max_iter):
#         price = black_scholes_call(S, K, T, sigma, r)
#         v = vega(S, K, T, sigma, r)
#         if v < 1e-8:
#             return None
#         diff = market_price - price
#         if abs(diff) < tol:
#             return sigma
#         sigma += diff / v
#     return sigma

# class Trader:
#     def run(self, state: TradingState):
#         result = {}
#         conversions = 0
#         traderDataDict = {'round_number': 1, 'historical_data': []}
#         if state.traderData:
#             try:
#                 traderDataDict = json.loads(state.traderData)
#             except:
#                 traderDataDict = {'round_number': 1, 'historical_data': []}

#         current_round = traderDataDict.get('round_number', 1)
#         traderDataDict['round_number'] = current_round + 1

#         St = None
#         if 'VOLCANIC_ROCK' in state.order_depths:
#             order_depth = state.order_depths['VOLCANIC_ROCK']
#             bids = order_depth.buy_orders
#             asks = order_depth.sell_orders
#             best_bid = max(bids.keys()) if bids else None
#             best_ask = min(asks.keys()) if asks else None
#             if best_bid is not None and best_ask is not None:
#                 St = (best_bid + best_ask) / 2

#         m_v_list = []
#         for product in state.order_depths:
#             if product.startswith('VOLCANIC_ROCK_VOUCHER_'):
#                 order_depth = state.order_depths[product]
#                 bids = order_depth.buy_orders
#                 asks = order_depth.sell_orders
#                 if not bids or not asks:
#                     continue
#                 best_bid = max(bids.keys())
#                 best_ask = min(asks.keys())
#                 Vt = (best_bid + best_ask) / 2
#                 strike_str = product.split('_')[-1]
#                 K = float(strike_str)
#                 TTE = 7 - (current_round - 1)
#                 if St is None or St <= 0 or TTE <= 0:
#                     continue
#                 m_t = math.log(K / St) / math.sqrt(TTE)
#                 sigma = compute_implied_vol(St, K, TTE, Vt)
#                 if sigma is not None:
#                     m_v_list.append((m_t, sigma))

#         a, b, c = 0, 0, 0
#         if len(m_v_list) >= 3:
#             X = []
#             y = []
#             for m, v in m_v_list:
#                 X.append([m**2, m, 1])
#                 y.append(v)
#             XT = [[X[j][i] for j in range(len(X))] for i in range(3)]
#             XTX = [
#                 [sum(XT[0][i] * XT[0][i] for i in range(len(XT[0]))), sum(XT[0][i] * XT[1][i] for i in range(len(XT[0]))), sum(XT[0][i] * XT[2][i] for i in range(len(XT[0])))],
#                 [sum(XT[1][i] * XT[0][i] for i in range(len(XT[0]))), sum(XT[1][i] * XT[1][i] for i in range(len(XT[0]))), sum(XT[1][i] * XT[2][i] for i in range(len(XT[0])))],
#                 [sum(XT[2][i] * XT[0][i] for i in range(len(XT[0]))), sum(XT[2][i] * XT[1][i] for i in range(len(XT[0]))), sum(XT[2][i] * XT[2][i] for i in range(len(XT[0])))]
#             ]
#             XTy = [
#                 sum(XT[0][i] * y[i] for i in range(len(y))),
#                 sum(XT[1][i] * y[i] for i in range(len(y))),
#                 sum(XT[2][i] * y[i] for i in range(len(y)))
#             ]
#             det = (XTX[0][0] * (XTX[1][1] * XTX[2][2] - XTX[1][2] * XTX[2][1]) -
#                    XTX[0][1] * (XTX[1][0] * XTX[2][2] - XTX[1][2] * XTX[2][0]) +
#                    XTX[0][2] * (XTX[1][0] * XTX[2][1] - XTX[1][1] * XTX[2][0]))
#             if det != 0:
#                 inv = [
#                     [
#                         (XTX[1][1] * XTX[2][2] - XTX[1][2] * XTX[2][1]) / det,
#                         (XTX[0][2] * XTX[2][1] - XTX[0][1] * XTX[2][2]) / det,
#                         (XTX[0][1] * XTX[1][2] - XTX[0][2] * XTX[1][1]) / det
#                     ],
#                     [
#                         (XTX[1][2] * XTX[2][0] - XTX[1][0] * XTX[2][2]) / det,
#                         (XTX[0][0] * XTX[2][2] - XTX[0][2] * XTX[2][0]) / det,
#                         (XTX[0][2] * XTX[1][0] - XTX[0][0] * XTX[1][2]) / det
#                     ],
#                     [
#                         (XTX[1][0] * XTX[2][1] - XTX[1][1] * XTX[2][0]) / det,
#                         (XTX[0][1] * XTX[2][0] - XTX[0][0] * XTX[2][1]) / det,
#                         (XTX[0][0] * XTX[1][1] - XTX[0][1] * XTX[1][0]) / det
#                     ]
#                 ]
#                 a = inv[0][0] * XTy[0] + inv[0][1] * XTy[1] + inv[0][2] * XTy[2]
#                 b = inv[1][0] * XTy[0] + inv[1][1] * XTy[1] + inv[1][2] * XTy[2]
#                 c = inv[2][0] * XTy[0] + inv[2][1] * XTy[1] + inv[2][2] * XTy[2]

#         for product in state.order_depths:
#             if product.startswith('VOLCANIC_ROCK_VOUCHER_'):
#                 order_depth = state.order_depths[product]
#                 orders = []
#                 bids = order_depth.buy_orders
#                 asks = order_depth.sell_orders
#                 if not bids or not asks or St is None:
#                     result[product] = orders
#                     continue
#                 best_bid = max(bids.keys())
#                 best_ask = min(asks.keys())
#                 Vt = (best_bid + best_ask) / 2
#                 strike_str = product.split('_')[-1]
#                 K = float(strike_str)
#                 TTE = 7 - (current_round - 1)
#                 if TTE <= 0 or St <= 0:
#                     result[product] = orders
#                     continue
#                 m_t = math.log(K / St) / math.sqrt(TTE)
#                 fitted_IV = a * m_t**2 + b * m_t + c
#                 fair_price = black_scholes_call(St, K, TTE, fitted_IV)
#                 current_position = state.position.get(product, 0)
                
#                 if Vt > fair_price + 1e-4:
#                     best_bid_volume = bids[best_bid]
#                     max_sell = current_position + 200
#                     sell_volume = min(best_bid_volume, max_sell)
#                     if sell_volume > 0:
#                         orders.append(Order(product, best_bid, -sell_volume))
#                 elif Vt < fair_price - 1e-4:
#                     best_ask_volume = asks[best_ask]
#                     max_buy = 200 - current_position
#                     buy_volume = min(-best_ask_volume, max_buy)
#                     if buy_volume > 0:
#                         orders.append(Order(product, best_ask, buy_volume))
#                 result[product] = orders

#         traderData = json.dumps(traderDataDict)
#         return result, conversions, traderData



import json
import math
from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List

def norm_pdf(x):
    return (1.0 / math.sqrt(2 * math.pi)) * math.exp(-x**2 / 2)

def norm_cdf(x):
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    sign = 1
    if x < 0:
        sign = -1
    x = abs(x) / math.sqrt(2.0)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t) * math.exp(-x*x)
    return 0.5 * (1.0 + sign * y)

def black_scholes_call(S, K, T, sigma, r=0):
    if T <= 0:
        return max(S - K, 0)
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    call_price = S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    return call_price

def vega(S, K, T, sigma, r=0):
    if T <= 0:
        return 0
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * sqrt_T)
    return S * norm_pdf(d1) * sqrt_T

def compute_implied_vol(S, K, T, market_price, r=0, max_iter=100, tol=1e-6):
    intrinsic = max(S - K, 0.0)
    if market_price < intrinsic - 1e-8 or market_price > S + 1e-8 or T <= 0:
        return None
    sigma = 0.5
    for _ in range(max_iter):
        price = black_scholes_call(S, K, T, sigma, r)
        v = vega(S, K, T, sigma, r)
        if v < 1e-8:
            return None
        diff = market_price - price
        if abs(diff) < tol:
            return sigma
        sigma += diff / v
    return sigma

class Trader:
    def run(self, state: TradingState):
        result = {}
        conversions = 0
        traderDataDict = {'round_number': 1, 'historical_data': [], 'prices': {'RAINFOREST_RESIN': [],'KELP': [],'SQUID_INK': []}}
        if state.traderData:
            try:
                traderDataDict = json.loads(state.traderData)
            except:
                traderDataDict = {'round_number': 1, 'historical_data': [], 'prices': {'RAINFOREST_RESIN': [],'KELP': [],'SQUID_INK': []}}

        for product in ["RAINFOREST_RESIN", "KELP", "SQUID_INK"]:
            if product in state.order_depths:
                order_depth = state.order_depths[product]
                buy_orders = order_depth.buy_orders
                sell_orders = order_depth.sell_orders
                best_bid = max(buy_orders.keys()) if buy_orders else None
                best_ask = min(sell_orders.keys()) if sell_orders else None
                if best_bid is not None and best_ask is not None:
                    mid_price = (best_bid + best_ask) / 2
                    traderDataDict['prices'][product].append(mid_price)
                    # Keep last 5 prices for moving average calculation
                    if len(traderDataDict['prices'][product]) > 5:
                        traderDataDict['prices'][product] = traderDataDict['prices'][product][-5:]

        for product in ["RAINFOREST_RESIN", "KELP", "SQUID_INK"]:
            if product not in state.order_depths:
                print(f"No order data available for {product}")
                continue
            
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            current_position = state.position.get(product, 0)
            
            if product == "RAINFOREST_RESIN":
                acceptable_price = 10000
                # Buy if best_ask < acceptable_price
                if order_depth.sell_orders:
                    best_ask = min(order_depth.sell_orders.keys())
                    best_ask_amount = order_depth.sell_orders[best_ask]
                    if best_ask < acceptable_price:
                        max_buy = 50 - current_position
                        amount = min(-best_ask_amount, max_buy)
                        if amount > 0:
                            print(f"BUY {product} {amount}x {best_ask}")
                            orders.append(Order(product, best_ask, amount))
                # Sell if best_bid > acceptable_price
                if order_depth.buy_orders:
                    best_bid = max(order_depth.buy_orders.keys())
                    best_bid_amount = order_depth.buy_orders[best_bid]
                    if best_bid > acceptable_price:
                        max_sell = current_position + 50
                        amount = min(best_bid_amount, max_sell)
                        if amount > 0:
                            print(f"SELL {product} {amount}x {best_bid}")
                            orders.append(Order(product, best_bid, -amount))
                result[product] = orders
            
            elif product == "KELP":
                # Calculate moving average for acceptable price
                price_history = traderDataDict['prices'].get(product, [])
                if len(price_history) >= 5:
                    ma = sum(price_history[-5:]) / 5
                else:
                    # Use current mid_price if available, else default to 2000
                    buy_orders = order_depth.buy_orders
                    sell_orders = order_depth.sell_orders
                    best_bid = max(buy_orders.keys()) if buy_orders else None
                    best_ask = min(sell_orders.keys()) if sell_orders else None
                    if best_bid and best_ask:
                        ma = (best_bid + best_ask) / 2
                    else:
                        ma = 2000
                # Adjust for upward trend
                acceptable_price = ma * 1.001
                # Buy if best_ask < acceptable_price
                if order_depth.sell_orders:
                    best_ask = min(order_depth.sell_orders.keys())
                    best_ask_amount = order_depth.sell_orders[best_ask]
                    if best_ask < acceptable_price:
                        max_buy = 50 - current_position
                        amount = min(-best_ask_amount, max_buy)
                        if amount > 0:
                            orders.append(Order(product, best_ask, amount))
                # Sell if best_bid > acceptable_price
                if order_depth.buy_orders:
                    best_bid = max(order_depth.buy_orders.keys())
                    best_bid_amount = order_depth.buy_orders[best_bid]
                    if best_bid > acceptable_price:
                        max_sell = current_position + 50
                        amount = min(best_bid_amount, max_sell)
                        if amount > 0:
                            orders.append(Order(product, best_bid, -amount))
                result[product] = orders
            
            elif product == "SQUID_INK":
                # Calculate moving average and thresholds
                price_history = traderDataDict['prices'].get(product, [])
                if len(price_history) >= 5:
                    ma = sum(price_history[-5:]) / 5
                else:
                    buy_orders = order_depth.buy_orders
                    sell_orders = order_depth.sell_orders
                    best_bid = max(buy_orders.keys()) if buy_orders else None
                    best_ask = min(sell_orders.keys()) if sell_orders else None
                    if best_bid and best_ask:
                        ma = (best_bid + best_ask) / 2
                    else:
                        ma = 2000
                threshold = 50
                lower_bound = ma - threshold
                upper_bound = ma + threshold
                # Buy if best_ask < lower_bound
                if order_depth.sell_orders:
                    best_ask = min(order_depth.sell_orders.keys())
                    best_ask_amount = order_depth.sell_orders[best_ask]
                    if best_ask < lower_bound:
                        max_buy = 50 - current_position
                        amount = min(-best_ask_amount, max_buy)
                        if amount > 0:
                            orders.append(Order(product, best_ask, amount))
                # Sell if best_bid > upper_bound
                if order_depth.buy_orders:
                    best_bid = max(order_depth.buy_orders.keys())
                    best_bid_amount = order_depth.buy_orders[best_bid]
                    if best_bid > upper_bound:
                        max_sell = current_position + 50
                        amount = min(best_bid_amount, max_sell)
                        if amount > 0:
                            orders.append(Order(product, best_bid, -amount))
                result[product] = orders

        current_round = traderDataDict.get('round_number', 1)
        traderDataDict['round_number'] = current_round + 1

        St = None
        if 'VOLCANIC_ROCK' in state.order_depths:
            order_depth = state.order_depths['VOLCANIC_ROCK']
            bids = order_depth.buy_orders
            asks = order_depth.sell_orders
            best_bid = max(bids.keys()) if bids else None
            best_ask = min(asks.keys()) if asks else None
            if best_bid is not None and best_ask is not None:
                St = (best_bid + best_ask) / 2

        m_v_list = []
        for product in state.order_depths:
            if product.startswith('VOLCANIC_ROCK_VOUCHER_'):
                order_depth = state.order_depths[product]
                bids = order_depth.buy_orders
                asks = order_depth.sell_orders
                if not bids or not asks:
                    continue
                best_bid = max(bids.keys())
                best_ask = min(asks.keys())
                Vt = (best_bid + best_ask) / 2
                strike_str = product.split('_')[-1]
                K = float(strike_str)
                TTE = 7 - (current_round - 1)
                if St is None or St <= 0 or TTE <= 0:
                    continue
                m_t = math.log(K / St) / math.sqrt(TTE)
                sigma = compute_implied_vol(St, K, TTE, Vt)
                if sigma is not None:
                    m_v_list.append((m_t, sigma))

        a, b, c = 0, 0, 0
        if len(m_v_list) >= 3:
            X = []
            y = []
            for m, v in m_v_list:
                X.append([m**2, m, 1])
                y.append(v)
            XT = [[X[j][i] for j in range(len(X))] for i in range(3)]
            XTX = [
                [sum(XT[0][i] * XT[0][i] for i in range(len(XT[0]))), sum(XT[0][i] * XT[1][i] for i in range(len(XT[0]))), sum(XT[0][i] * XT[2][i] for i in range(len(XT[0])))],
                [sum(XT[1][i] * XT[0][i] for i in range(len(XT[0]))), sum(XT[1][i] * XT[1][i] for i in range(len(XT[0]))), sum(XT[1][i] * XT[2][i] for i in range(len(XT[0])))],
                [sum(XT[2][i] * XT[0][i] for i in range(len(XT[0]))), sum(XT[2][i] * XT[1][i] for i in range(len(XT[0]))), sum(XT[2][i] * XT[2][i] for i in range(len(XT[0])))]
            ]
            XTy = [
                sum(XT[0][i] * y[i] for i in range(len(y))),
                sum(XT[1][i] * y[i] for i in range(len(y))),
                sum(XT[2][i] * y[i] for i in range(len(y)))
            ]
            det = (XTX[0][0] * (XTX[1][1] * XTX[2][2] - XTX[1][2] * XTX[2][1]) -
                   XTX[0][1] * (XTX[1][0] * XTX[2][2] - XTX[1][2] * XTX[2][0]) +
                   XTX[0][2] * (XTX[1][0] * XTX[2][1] - XTX[1][1] * XTX[2][0]))
            if det != 0:
                inv = [
                    [
                        (XTX[1][1] * XTX[2][2] - XTX[1][2] * XTX[2][1]) / det,
                        (XTX[0][2] * XTX[2][1] - XTX[0][1] * XTX[2][2]) / det,
                        (XTX[0][1] * XTX[1][2] - XTX[0][2] * XTX[1][1]) / det
                    ],
                    [
                        (XTX[1][2] * XTX[2][0] - XTX[1][0] * XTX[2][2]) / det,
                        (XTX[0][0] * XTX[2][2] - XTX[0][2] * XTX[2][0]) / det,
                        (XTX[0][2] * XTX[1][0] - XTX[0][0] * XTX[1][2]) / det
                    ],
                    [
                        (XTX[1][0] * XTX[2][1] - XTX[1][1] * XTX[2][0]) / det,
                        (XTX[0][1] * XTX[2][0] - XTX[0][0] * XTX[2][1]) / det,
                        (XTX[0][0] * XTX[1][1] - XTX[0][1] * XTX[1][0]) / det
                    ]
                ]
                a = inv[0][0] * XTy[0] + inv[0][1] * XTy[1] + inv[0][2] * XTy[2]
                b = inv[1][0] * XTy[0] + inv[1][1] * XTy[1] + inv[1][2] * XTy[2]
                c = inv[2][0] * XTy[0] + inv[2][1] * XTy[1] + inv[2][2] * XTy[2]

        for product in state.order_depths:
            if product.startswith('VOLCANIC_ROCK_VOUCHER_'):
                order_depth = state.order_depths[product]
                orders = []
                bids = order_depth.buy_orders
                asks = order_depth.sell_orders
                if not bids or not asks or St is None:
                    result[product] = orders
                    continue
                best_bid = max(bids.keys())
                best_ask = min(asks.keys())
                Vt = (best_bid + best_ask) / 2
                strike_str = product.split('_')[-1]
                K = float(strike_str)
                TTE = 7 - (current_round - 1)
                if TTE <= 0 or St <= 0:
                    result[product] = orders
                    continue
                m_t = math.log(K / St) / math.sqrt(TTE)
                fitted_IV = a * m_t**2 + b * m_t + c
                fair_price = black_scholes_call(St, K, TTE, fitted_IV)
                current_position = state.position.get(product, 0)
                
                if Vt > fair_price + 1e-4:
                    best_bid_volume = bids[best_bid]
                    max_sell = current_position + 200
                    sell_volume = min(best_bid_volume, max_sell)
                    if sell_volume > 0:
                        orders.append(Order(product, best_bid, -sell_volume))
                elif Vt < fair_price - 1e-4:
                    best_ask_volume = asks[best_ask]
                    max_buy = 200 - current_position
                    buy_volume = min(-best_ask_volume, max_buy)
                    if buy_volume > 0:
                        orders.append(Order(product, best_ask, buy_volume))
                result[product] = orders

        traderData = json.dumps(traderDataDict)
        return result, conversions, traderData