from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string

class Trader:
    
    def run(self, state: TradingState):
        # Only method required. It takes all buy and sell orders for all symbols as an input, and outputs a list of orders to be sent
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))
        
        result = {}
        
        # Define the acceptable price for each product
        acceptable_prices = {
            "RAINFOREST_RESIN": 10,  # Adjust the acceptable price for RAINFOREST_RESIN
            "KELP": 15  # Adjust the acceptable price for KELP
        }

        # Loop through specific products ("RAINFOREST_RESIN" and "KELP")
        for product in ["RAINFOREST_RESIN", "KELP"]:
            if product not in state.order_depths:
                print(f"No order data available for {product}")
                continue
            
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            acceptable_price = acceptable_prices.get(product, 10)  # Default to 10 if product not found
            print(f"Acceptable price for {product}: " + str(acceptable_price))
            print(f"Buy Order depth for {product}: " + str(len(order_depth.buy_orders)) + ", Sell order depth: " + str(len(order_depth.sell_orders)))

            # Check sell orders
            if len(order_depth.sell_orders) != 0:
                best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                if int(best_ask) < acceptable_price:
                    print(f"BUY {product}: {str(-best_ask_amount)}x {best_ask}")
                    orders.append(Order(product, best_ask, -best_ask_amount))

            # Check buy orders
            if len(order_depth.buy_orders) != 0:
                best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                if int(best_bid) > acceptable_price:
                    print(f"SELL {product}: {str(best_bid_amount)}x {best_bid}")
                    orders.append(Order(product, best_bid, -best_bid_amount))

            # Save the orders for the current product
            result[product] = orders

        # The traderData can be updated to reflect the state
        traderData = "SAMPLE"  # String value holding Trader state data required.
        
        conversions = 1
        return result, conversions, traderData
