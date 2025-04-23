# ~770 algorithm

from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import json

class Trader:
    
    def run(self, state: TradingState):
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))
        
        result = {}
        
        # Parse traderData to get historical prices
        if state.traderData:
            trader_data = json.loads(state.traderData)
        else:
            trader_data = {
                'prices': {
                    'RAINFOREST_RESIN': [],
                    'KELP': [],
                    'SQUID_INK': []
                }
            }
        
        # Update historical mid_prices for each product
        for product in ["RAINFOREST_RESIN", "KELP", "SQUID_INK"]:
            if product in state.order_depths:
                order_depth = state.order_depths[product]
                buy_orders = order_depth.buy_orders
                sell_orders = order_depth.sell_orders
                best_bid = max(buy_orders.keys()) if buy_orders else None
                best_ask = min(sell_orders.keys()) if sell_orders else None
                if best_bid is not None and best_ask is not None:
                    mid_price = (best_bid + best_ask) / 2
                    trader_data['prices'][product].append(mid_price)
                    # Keep last 5 prices for moving average calculation
                    if len(trader_data['prices'][product]) > 5:
                        trader_data['prices'][product] = trader_data['prices'][product][-5:]
        
        # Process each product to generate orders
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
                price_history = trader_data['prices'].get(product, [])
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
                price_history = trader_data['prices'].get(product, [])
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
        
        # Serialize trader_data for next iteration
        traderData = json.dumps(trader_data)
        
        conversions = 1
        return result, conversions, traderData


# [-996,924] algorithm

# from datamodel import OrderDepth, UserId, TradingState, Order
# from typing import List
# import json

# class Trader:
    
#     def run(self, state: TradingState):
#         print("traderData: " + state.traderData)
#         print("Observations: " + str(state.observations))
        
#         result = {}
        
#         # Parse traderData to get historical prices
#         if state.traderData:
#             trader_data = json.loads(state.traderData)
#         else:
#             trader_data = {
#                 'prices': {
#                     'RAINFOREST_RESIN': [],
#                     'KELP': [],
#                     'SQUID_INK': []
#                 }
#             }
        
#         # Update historical mid_prices for each product
#         for product in ["RAINFOREST_RESIN", "KELP", "SQUID_INK"]:
#             if product in state.order_depths:
#                 order_depth = state.order_depths[product]
#                 buy_orders = order_depth.buy_orders
#                 sell_orders = order_depth.sell_orders
#                 best_bid = max(buy_orders.keys()) if buy_orders else None
#                 best_ask = min(sell_orders.keys()) if sell_orders else None
#                 if best_bid is not None and best_ask is not None:
#                     mid_price = (best_bid + best_ask) / 2
#                     trader_data['prices'][product].append(mid_price)
#                     # Keep last 5 prices for moving average calculation
#                     if len(trader_data['prices'][product]) > 5:
#                         trader_data['prices'][product] = trader_data['prices'][product][-5:]
        
#         # Process each product to generate orders
#         for product in ["RAINFOREST_RESIN", "KELP", "SQUID_INK"]:
#             if product not in state.order_depths:
#                 print(f"No order data available for {product}")
#                 continue
            
#             order_depth: OrderDepth = state.order_depths[product]
#             orders: List[Order] = []
#             current_position = state.position.get(product, 0)
            
#             if product == "RAINFOREST_RESIN":
#                 acceptable_price = 10000
#                 # Buy if best_ask < acceptable_price
#                 if order_depth.sell_orders:
#                     best_ask = min(order_depth.sell_orders.keys())
#                     best_ask_amount = order_depth.sell_orders[best_ask]
#                     if best_ask < acceptable_price:
#                         max_buy = 50 - current_position
#                         amount = min(-best_ask_amount, max_buy)
#                         if amount > 0:
#                             print(f"BUY {product} {amount}x {best_ask}")
#                             orders.append(Order(product, best_ask, amount))
#                 # Sell if best_bid > acceptable_price
#                 if order_depth.buy_orders:
#                     best_bid = max(order_depth.buy_orders.keys())
#                     best_bid_amount = order_depth.buy_orders[best_bid]
#                     if best_bid > acceptable_price:
#                         max_sell = current_position + 50
#                         amount = min(best_bid_amount, max_sell)
#                         if amount > 0:
#                             print(f"SELL {product} {amount}x {best_bid}")
#                             orders.append(Order(product, best_bid, -amount))
#                 result[product] = orders
            
#             elif product == "KELP":
#                 price_history = trader_data['prices'].get(product, [])
#                 if price_history:
#                     ma = sum(price_history) / len(price_history)
#                 else:
#                     buy_orders = order_depth.buy_orders
#                     sell_orders = order_depth.sell_orders
#                     best_bid = max(buy_orders.keys()) if buy_orders else None
#                     best_ask = min(sell_orders.keys()) if sell_orders else None
#                     if best_bid and best_ask:
#                         ma = (best_bid + best_ask) / 2
#                     else:
#                         ma = 2000
#                 # Adjust for upward trend
#                 acceptable_price = ma * 1.001
#                 # Buy if best_ask < acceptable_price
#                 if order_depth.sell_orders:
#                     best_ask = min(order_depth.sell_orders.keys())
#                     best_ask_amount = order_depth.sell_orders[best_ask]
#                     if best_ask < acceptable_price:
#                         max_buy = 50 - current_position
#                         amount = min(-best_ask_amount, max_buy)
#                         if amount > 0:
#                             orders.append(Order(product, best_ask, amount))
#                 # Sell if best_bid > acceptable_price
#                 if order_depth.buy_orders:
#                     best_bid = max(order_depth.buy_orders.keys())
#                     best_bid_amount = order_depth.buy_orders[best_bid]
#                     if best_bid > acceptable_price:
#                         max_sell = current_position + 50
#                         amount = min(best_bid_amount, max_sell)
#                         if amount > 0:
#                             orders.append(Order(product, best_bid, -amount))
#                 result[product] = orders
            
#             elif product == "SQUID_INK":
#                 price_history = trader_data['prices'].get(product, [])
#                 if price_history:
#                     ma = sum(price_history) / len(price_history)
#                     if len(price_history) >= 2:
#                         # More responsive volatility measurement
#                         deviations = [abs(p - ma) for p in price_history]
#                         mad = sum(deviations) / len(deviations)
#                         # Tighter threshold (1.5x MAD instead of 2x)
#                         threshold = 1.5 * mad
#                     else:
#                         threshold = 40  # Slightly tighter default
#                 else:
#                     best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
#                     best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
#                     ma = (best_bid + best_ask) / 2 if best_bid and best_ask else 2000
#                     threshold = 40
                
#                 lower_bound = ma - threshold
#                 upper_bound = ma + threshold
                
#                 # Aggressive buying: take all sell orders below lower bound
#                 max_buy = 50 - current_position
#                 if max_buy > 0:
#                     # Sort sell orders by price ascending
#                     asks = sorted(order_depth.sell_orders.items())
#                     for ask_price, ask_amount in asks:
#                         if ask_price > lower_bound:
#                             break
#                         # Take 75% of available volume instead of min()
#                         take_amount = min(int(-ask_amount * 0.75), max_buy)
#                         if take_amount > 0:
#                             orders.append(Order(product, ask_price, take_amount))
#                             max_buy -= take_amount
                
#                 # Aggressive selling: take all buy orders above upper bound
#                 max_sell = current_position + 50
#                 if max_sell > 0:
#                     # Sort buy orders by price descending
#                     bids = sorted(order_depth.buy_orders.items(), reverse=True)
#                     for bid_price, bid_amount in bids:
#                         if bid_price < upper_bound:
#                             break
#                         # Take 75% of available volume instead of min()
#                         take_amount = min(int(bid_amount * 0.75), max_sell)
#                         if take_amount > 0:
#                             orders.append(Order(product, bid_price, -take_amount))
#                             max_sell -= take_amount
                
#                 result[product] = orders
        
#         # Serialize trader_data for next iteration
#         traderData = json.dumps(trader_data)
        
#         conversions = 1
#         return result, conversions, traderData

