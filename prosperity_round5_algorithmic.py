from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import json
import math

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
    def __init__(self):
        self.position_limit = 75
        self.conversion_limit = 10
        self.storage_cost = 0.1
        self.learning_rate = 0.001
        self.margin_buy = 2.0
        self.margin_sell = 0.5
        self.N = 5  # Number of timestamps for pressure history
        self.k = 0.1  # Adjustment factor for fair value based on pressure

    def calculate_buy_arbitrage(self, order_depth: OrderDepth, P_sell_conv: float, current_position: int) -> tuple[float, List[Order], int]:
        sell_orders = order_depth.sell_orders
        prices = sorted([p for p in sell_orders.keys() if p < P_sell_conv])
        total_profit = 0
        total_volume = 0
        orders = []
        
        for price in prices:
            volume_available = -sell_orders[price]
            remaining_capacity = min(self.conversion_limit - total_volume, self.position_limit - (current_position + total_volume))
            volume_to_buy = min(volume_available, remaining_capacity)
            if volume_to_buy > 0:
                profit = (P_sell_conv - price) * volume_to_buy
                total_profit += profit
                total_volume += volume_to_buy
                orders.append(Order("MAGNIFICENT_MACARONS", price, volume_to_buy))
                if total_volume >= self.conversion_limit:
                    break
        return total_profit, orders, total_volume

    def calculate_sell_arbitrage(self, order_depth: OrderDepth, P_buy_conv: float, current_position: int) -> tuple[float, List[Order], int]:
        buy_orders = order_depth.buy_orders
        prices = sorted([p for p in buy_orders.keys() if p > P_buy_conv], reverse=True)
        total_profit = 0
        total_volume = 0
        orders = []
        
        for price in prices:
            volume_available = buy_orders[price]
            remaining_capacity = min(self.conversion_limit - total_volume, self.position_limit + (current_position - total_volume))
            volume_to_sell = min(volume_available, remaining_capacity)
            if volume_to_sell > 0:
                profit = (price - P_buy_conv) * volume_to_sell
                total_profit += profit
                total_volume += volume_to_sell
                orders.append(Order("MAGNIFICENT_MACARONS", price, -volume_to_sell))
                if total_volume >= self.conversion_limit:
                    break
        return total_profit, orders, total_volume

    def run(self, state: TradingState) -> tuple[dict, int, str]:
        result = {}
        conversions = 0

        current_position = state.position.get("MAGNIFICENT_MACARONS", 0)
        
        if "MAGNIFICENT_MACARONS" not in state.order_depths or "MAGNIFICENT_MACARONS" not in state.observations.conversionObservations:
            return result, conversions, state.traderData
        
        order_depth = state.order_depths["MAGNIFICENT_MACARONS"]
        conversion = state.observations.conversionObservations["MAGNIFICENT_MACARONS"]

        P_conv_bid = conversion.bidPrice
        P_conv_ask = conversion.askPrice
        TF = conversion.transportFees
        ET = conversion.exportTariff
        IT = conversion.importTariff
        P_sell_conv = P_conv_bid - TF - ET
        P_buy_conv = P_conv_ask + TF + IT

        traderData_dict = json.loads(state.traderData) if state.traderData else {"a": 0, "b": 0, "c": 0}
        a, b, c = traderData_dict["a"], traderData_dict["b"], traderData_dict["c"]

        buy_prices = sorted(order_depth.buy_orders.keys(), reverse=True)
        sell_prices = sorted(order_depth.sell_orders.keys())
        mid_price = None
        if buy_prices and sell_prices:
            best_bid = buy_prices[0]
            best_ask = sell_prices[0]
            mid_price = (best_bid + best_ask) / 2

        sugar_price = conversion.sugarPrice
        sunlight_index = conversion.sunlightIndex
        if mid_price is not None:
            predicted = a * sugar_price + b * sunlight_index + c
            error = mid_price - predicted
            a += self.learning_rate * error * sugar_price
            b += self.learning_rate * error * sunlight_index
            c += self.learning_rate * error

        fair_value = a * sugar_price + b * sunlight_index + c

        buy_profit, buy_orders, buy_volume = self.calculate_buy_arbitrage(order_depth, P_sell_conv, current_position)
        sell_profit, sell_orders, sell_volume = self.calculate_sell_arbitrage(order_depth, P_buy_conv, current_position)

        if buy_profit > sell_profit and buy_profit > 0:
            result["MAGNIFICENT_MACARONS"] = buy_orders
            conversions = -buy_volume
        elif sell_profit > 0:
            result["MAGNIFICENT_MACARONS"] = sell_orders
            conversions = sell_volume
        else:
            result["MAGNIFICENT_MACARONS"] = []

        if 'pressure_history' not in traderData_dict:
            traderData_dict['pressure_history'] = {product: [] for product in ["MAGNIFICENT_MACARONS", "RAINFOREST_RESIN", "KELP", "SQUID_INK"]}
        
        for product in ["MAGNIFICENT_MACARONS", "RAINFOREST_RESIN", "KELP", "SQUID_INK"]:
            if product in state.own_trades:
                total_pressure = sum(-trade.quantity for trade in state.own_trades[product])
            else:
                total_pressure = 0
            traderData_dict['pressure_history'][product].append(total_pressure)
            if len(traderData_dict['pressure_history'][product]) > self.N:
                traderData_dict['pressure_history'][product] = traderData_dict['pressure_history'][product][-self.N:]
        
        recent_pressure = {product: sum(traderData_dict['pressure_history'][product]) for product in ["MAGNIFICENT_MACARONS", "RAINFOREST_RESIN", "KELP", "SQUID_INK"]}
        
        fair_value_adjusted = fair_value + self.k * recent_pressure.get("MAGNIFICENT_MACARONS", 0)

        if buy_prices and sell_prices:
            if best_ask < fair_value_adjusted - self.margin_buy and current_position < self.position_limit:
                max_buy = self.position_limit - current_position
                buy_quantity = min(10, max_buy)
                volume_available = -order_depth.sell_orders.get(best_ask, 0)
                buy_quantity = min(buy_quantity, volume_available)
                if buy_quantity > 0:
                    result["MAGNIFICENT_MACARONS"].append(Order("MAGNIFICENT_MACARONS", best_ask, buy_quantity))

            if best_bid > fair_value_adjusted + self.margin_sell and current_position > -self.position_limit:
                max_sell = self.position_limit + current_position
                sell_quantity = min(10, max_sell)
                volume_available = order_depth.buy_orders.get(best_bid, 0)
                sell_quantity = min(sell_quantity, volume_available)
                if sell_quantity > 0:
                    result["MAGNIFICENT_MACARONS"].append(Order("MAGNIFICENT_MACARONS", best_bid, -sell_quantity))

        traderData_dict["a"] = a
        traderData_dict["b"] = b
        traderData_dict["c"] = c

        traderDataDict = {'round_number': 1, 'historical_data': [], 'prices': {'RAINFOREST_RESIN': [], 'KELP': [], 'SQUID_INK': []}}
        if state.traderData:
            try:
                traderDataDict = json.loads(state.traderData)
            except:
                traderDataDict = {'round_number': 1, 'historical_data': [], 'prices': {'RAINFOREST_RESIN': [], 'KELP': [], 'SQUID_INK': []}}

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
                    if len(traderDataDict['prices'][product]) > 5:
                        traderDataDict['prices'][product] = traderDataDict['prices'][product][-5:]

        for product in ["RAINFOREST_RESIN", "KELP", "SQUID_INK"]:
            if product not in state.order_depths:
                print(f"No order data available for {product}")
                continue
            
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            current_position = state.position.get(product, 0)
            price_history = traderDataDict['prices'].get(product, [])
            
            if product == "RAINFOREST_RESIN":
                acceptable_price = 10000 + self.k * recent_pressure.get(product, 0)
                if order_depth.sell_orders:
                    best_ask = min(order_depth.sell_orders.keys())
                    best_ask_amount = order_depth.sell_orders[best_ask]
                    if best_ask < acceptable_price:
                        max_buy = 50 - current_position
                        amount = min(-best_ask_amount, max_buy)
                        if amount > 0:
                            print(f"BUY {product} {amount}x {best_ask}")
                            orders.append(Order(product, best_ask, amount))
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
                acceptable_price = ma * 1.001 + self.k * recent_pressure.get(product, 0)
                if order_depth.sell_orders:
                    best_ask = min(order_depth.sell_orders.keys())
                    best_ask_amount = order_depth.sell_orders[best_ask]
                    if best_ask < acceptable_price:
                        max_buy = 50 - current_position
                        amount = min(-best_ask_amount, max_buy)
                        if amount > 0:
                            orders.append(Order(product, best_ask, amount))
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
                lower_bound = ma - threshold + self.k * recent_pressure.get(product, 0)
                upper_bound = ma + threshold + self.k * recent_pressure.get(product, 0)
                if order_depth.sell_orders:
                    best_ask = min(order_depth.sell_orders.keys())
                    best_ask_amount = order_depth.sell_orders[best_ask]
                    if best_ask < lower_bound:
                        max_buy = 50 - current_position
                        amount = min(-best_ask_amount, max_buy)
                        if amount > 0:
                            orders.append(Order(product, best_ask, amount))
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

        traderData = json.dumps({**traderData_dict, **traderDataDict})
        return result, conversions, traderData