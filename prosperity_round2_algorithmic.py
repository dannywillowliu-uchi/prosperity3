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
        
#         # Update historical mid_prices for Round 1 products
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
#                     if len(trader_data['prices'][product]) > 5:
#                         trader_data['prices'][product] = trader_data['prices'][product][-5:]
        
#         # Update historical mid_prices for Round 2 products
#         new_products = ["CROISSANTS", "JAMS", "DJEMBE", "PICNIC_BASKET1", "PICNIC_BASKET2"]
#         for product in new_products:
#             if product in state.order_depths:
#                 order_depth = state.order_depths[product]
#                 buy_orders = order_depth.buy_orders
#                 sell_orders = order_depth.sell_orders
#                 best_bid = max(buy_orders.keys()) if buy_orders else None
#                 best_ask = min(sell_orders.keys()) if sell_orders else None
#                 if best_bid is not None and best_ask is not None:
#                     mid_price = (best_bid + best_ask) / 2
#                     if product not in trader_data['prices']:
#                         trader_data['prices'][product] = []
#                     trader_data['prices'][product].append(mid_price)
#                     if len(trader_data['prices'][product]) > 5:
#                         trader_data['prices'][product] = trader_data['prices'][product][-5:]
        
#         # Process each Round 1 product to generate orders
#         for product in ["RAINFOREST_RESIN", "KELP", "SQUID_INK"]:
#             if product not in state.order_depths:
#                 print(f"No order data available for {product}")
#                 continue
            
#             order_depth: OrderDepth = state.order_depths[product]
#             orders: List[Order] = []
#             current_position = state.position.get(product, 0)
            
#             if product == "RAINFOREST_RESIN":
#                 acceptable_price = 10000
#                 if order_depth.sell_orders:
#                     best_ask = min(order_depth.sell_orders.keys())
#                     best_ask_amount = order_depth.sell_orders[best_ask]
#                     if best_ask < acceptable_price:
#                         max_buy = 50 - current_position
#                         amount = min(-best_ask_amount, max_buy)
#                         if amount > 0:
#                             print(f"BUY {product} {amount}x {best_ask}")
#                             orders.append(Order(product, best_ask, amount))
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
#                 if len(price_history) >= 5:
#                     ma = sum(price_history[-5:]) / 5
#                 else:
#                     buy_orders = order_depth.buy_orders
#                     sell_orders = order_depth.sell_orders
#                     best_bid = max(buy_orders.keys()) if buy_orders else None
#                     best_ask = min(sell_orders.keys()) if sell_orders else None
#                     if best_bid and best_ask:
#                         ma = (best_bid + best_ask) / 2
#                     else:
#                         ma = 2000
#                 acceptable_price = ma * 1.001
#                 if order_depth.sell_orders:
#                     best_ask = min(order_depth.sell_orders.keys())
#                     best_ask_amount = order_depth.sell_orders[best_ask]
#                     if best_ask < acceptable_price:
#                         max_buy = 50 - current_position
#                         amount = min(-best_ask_amount, max_buy)
#                         if amount > 0:
#                             orders.append(Order(product, best_ask, amount))
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
#                 if len(price_history) >= 5:
#                     ma = sum(price_history[-5:]) / 5
#                 else:
#                     buy_orders = order_depth.buy_orders
#                     sell_orders = order_depth.sell_orders
#                     best_bid = max(buy_orders.keys()) if buy_orders else None
#                     best_ask = min(sell_orders.keys()) if sell_orders else None
#                     if best_bid and best_ask:
#                         ma = (best_bid + best_ask) / 2
#                     else:
#                         ma = 2000
#                 threshold = 50
#                 lower_bound = ma - threshold
#                 upper_bound = ma + threshold
#                 if order_depth.sell_orders:
#                     best_ask = min(order_depth.sell_orders.keys())
#                     best_ask_amount = order_depth.sell_orders[best_ask]
#                     if best_ask < lower_bound:
#                         max_buy = 50 - current_position
#                         amount = min(-best_ask_amount, max_buy)
#                         if amount > 0:
#                             orders.append(Order(product, best_ask, amount))
#                 if order_depth.buy_orders:
#                     best_bid = max(order_depth.buy_orders.keys())
#                     best_bid_amount = order_depth.buy_orders[best_bid]
#                     if best_bid > upper_bound:
#                         max_sell = current_position + 50
#                         amount = min(best_bid_amount, max_sell)
#                         if amount > 0:
#                             orders.append(Order(product, best_bid, -amount))
#                 result[product] = orders
        
#         # Process Round 2 products
#         for product in new_products:
#             if product not in state.order_depths:
#                 continue
            
#             order_depth = state.order_depths[product]
#             orders = []
#             current_position = state.position.get(product, 0)
            
#             if product in ["CROISSANTS", "JAMS", "DJEMBE"]:
#                 if product == "CROISSANTS":
#                     position_limit = 250
#                 elif product == "JAMS":
#                     position_limit = 350
#                 else:  # DJEMBE
#                     position_limit = 60
                
#                 price_history = trader_data['prices'].get(product, [])
#                 if len(price_history) >= 5:
#                     ma = sum(price_history[-5:]) / 5
#                 else:
#                     buy_orders = order_depth.buy_orders
#                     sell_orders = order_depth.sell_orders
#                     best_bid = max(buy_orders.keys()) if buy_orders else None
#                     best_ask = min(sell_orders.keys()) if sell_orders else None
#                     if best_bid and best_ask:
#                         ma = (best_bid + best_ask) / 2
#                     else:
#                         # Not enough data, skip
#                         result[product] = []
#                         continue
                
#                 threshold = 50
#                 lower_bound = ma - threshold
#                 upper_bound = ma + threshold
                
#                 # Buy if best_ask < lower_bound
#                 if order_depth.sell_orders:
#                     best_ask = min(order_depth.sell_orders.keys())
#                     best_ask_amount = order_depth.sell_orders[best_ask]
#                     if best_ask < lower_bound:
#                         max_buy = position_limit - current_position
#                         amount = min(-best_ask_amount, max_buy)
#                         if amount > 0:
#                             orders.append(Order(product, best_ask, amount))
                
#                 # Sell if best_bid > upper_bound
#                 if order_depth.buy_orders:
#                     best_bid = max(order_depth.buy_orders.keys())
#                     best_bid_amount = order_depth.buy_orders[best_bid]
#                     if best_bid > upper_bound:
#                         max_sell = current_position + position_limit
#                         amount = min(best_bid_amount, max_sell)
#                         if amount > 0:
#                             orders.append(Order(product, best_bid, -amount))
                
#                 result[product] = orders
            
#             elif product in ["PICNIC_BASKET1", "PICNIC_BASKET2"]:
#                 # Determine components and position limit
#                 if product == "PICNIC_BASKET1":
#                     components = {'CROISSANTS': 6, 'JAMS': 3, 'DJEMBE': 1}
#                     position_limit = 60
#                 else:
#                     components = {'CROISSANTS': 4, 'JAMS': 2}
#                     position_limit = 100
                
#                 # Check if all components have valid prices
#                 valid = True
#                 synthetic_price = 0
#                 for comp, qty in components.items():
#                     if comp not in trader_data['prices'] or len(trader_data['prices'][comp]) == 0:
#                         valid = False
#                         break
#                     comp_mid = trader_data['prices'][comp][-1]
#                     synthetic_price += qty * comp_mid
                
#                 if not valid:
#                     result[product] = []
#                     continue
                
#                 # Get current basket mid price
#                 buy_orders = order_depth.buy_orders
#                 sell_orders = order_depth.sell_orders
#                 best_bid = max(buy_orders.keys()) if buy_orders else None
#                 best_ask = min(sell_orders.keys()) if sell_orders else None
                
#                 if not best_bid or not best_ask:
#                     result[product] = []
#                     continue
                
#                 basket_mid = (best_bid + best_ask) / 2
#                 spread = basket_mid - synthetic_price
                
#                 # Set threshold based on product
#                 threshold = 100 if product == "PICNIC_BASKET1" else 50
                
#                 # Generate orders based on spread
#                 if spread > threshold:
#                     # Sell the basket
#                     max_sell = current_position + position_limit
#                     amount = min(buy_orders[best_bid], max_sell)
#                     if amount > 0:
#                         orders.append(Order(product, best_bid, -amount))
#                 elif spread < -threshold:
#                     # Buy the basket
#                     max_buy = position_limit - current_position
#                     amount = min(-sell_orders[best_ask], max_buy)
#                     if amount > 0:
#                         orders.append(Order(product, best_ask, amount))
                
#                 result[product] = orders
        
#         # Serialize trader_data for next iteration
#         traderData = json.dumps(trader_data)
        
#         conversions = 1
#         return result, conversions, traderData

