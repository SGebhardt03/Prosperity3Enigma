import json
import jsonpickle
from typing import Any

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()


class Trader:
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result = {}
        conversions = 0
        trader_data = ""

        limits = {
            "RAINFOREST_RESIN": 50,
            "KELP": 50
        }

        # create / load logger
        # the trader Data is a dict with objects
        trader_data_object = {}
        try:
            trader_data_object = jsonpickle.decode(state.traderData)
        except:
            # do not logger.print here!!!
            print("trader Data could not be fetched, assuming it is empty")
        logger = Logger()
        if ("logger" in trader_data_object.keys()):
            logger = trader_data_object["logger"]

        ### --------------------------actual trading logic------------------------------------------------
        for product in state.order_depths:
            # Implementation only for RAINFOREST_RESIN bc its price is stable
            orders: List[Order] = []
            buy_power = limits[product]
            sell_power = limits[product]
            if product in state.position:
                buy_power = 50 - int(state.position[product])
                sell_power = 50 + int(state.position[product])
            if product == "RAINFOREST_RESIN":
                # Prüfe, ob es überhaupt eine Position gibt, sonst stürzt Code ab :(
                orders.append(Order(product, 9999, buy_power))
                orders.append(Order(product, 10001, -sell_power))

            else:
                # Strategy: always place orders for the best order in the order book depending on acceptable price
                order_depth: OrderDepth = state.order_depths[product]
                acceptable_price = 10;  # Participant should calculate this value
                logger.print("Acceptable price : " + str(acceptable_price))
                logger.print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(
                    len(order_depth.sell_orders)))

                if len(order_depth.sell_orders) != 0:
                    best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                    if int(best_ask) < acceptable_price:
                        print("BUY", str(-best_ask_amount) + "x", best_ask)
                        orders.append(Order(product, best_ask, min((buy_power, abs(best_ask_amount)))))

                if len(order_depth.buy_orders) != 0:
                    best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                    if int(best_bid) > acceptable_price:
                        logger.print("SELL", str(best_bid_amount) + "x", best_bid)
                        orders.append(Order(product, best_bid, -min(sell_power, abs(best_ask_amount))))

            # remove 0 quantity orders
            orders = list(filter(lambda x: x.quantity != 0, orders))

            result[product] = orders

        # --------------------------return value ajustments----------------------------------------------------
        # this will become important later in the game
        conversions = 1

        logger.flush(state, result, conversions, state.traderData)
        # backup the recent version of the logger object
        trader_data_object["logger"] = logger
        traderData = jsonpickle.encode(trader_data_object)

        return result, conversions, str(traderData)