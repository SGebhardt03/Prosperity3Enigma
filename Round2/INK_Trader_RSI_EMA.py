import json
import math

import jsonpickle
import numpy as np
from typing import Any, List

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState


class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"


LIMITS = {
    "RAINFOREST_RESIN": 50,
    "KELP": 50,
    "SQUID_INK": 50
}

PARAMS = {
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,
        "take_width": 1,
        "clear_width": 0.5,
        "volume_limit": 0,
    },
    Product.KELP: {
        "delta_t": 10,
        "take_width": 1,
        "clear_width": 0.5,
        "prevent_adverse": True,
        "adverse_volume": 15,
        "reversion_beta": 0,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
        "volume_limit": 0
    },
    Product.SQUID_INK: {
        "delta_t": 5,
        "delta_rsi": 25,
        "delta_ema": 20,
        "delta_ema2": 50,
        "take_width": 1,
        "clear_width": 0.5,
        "prevent_adverse": True,
        "adverse_volume": 15,
        "reversion_beta": 0,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
        "volume_limit": 0
    }
}


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

    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params

        self.LIMIT = {
            Product.RAINFOREST_RESIN: 50,
            Product.KELP: 50,
            Product.SQUID_INK: 50
        }

    def market_price(self, product, state):
        # Calculates market price as mean of orders
        buy_orders = state.order_depths[product].buy_orders
        sell_orders = state.order_depths[product].sell_orders
        mean = 0
        weights = 0
        for prize in buy_orders:
            mean += prize * np.abs(buy_orders[prize])
            weights += np.abs(buy_orders[prize])
        for prize in sell_orders:
            mean += prize * np.abs(sell_orders[prize])
            weights += np.abs(sell_orders[prize])
        if weights != 0:
            mean = mean / weights
            return int(mean)
        return 0

    def market_mid_price(self, product, state):
        # Calculates market price as mean of best buy and sell
        buy_orders = state.order_depths[product].buy_orders
        sell_orders = state.order_depths[product].sell_orders
        best_buy = list(buy_orders.items())[0][0]
        best_sell = list(sell_orders.items())[0][0]

        return (best_buy+best_sell)/2

    def update_averager(self, average_value, old_average: list):
        new_average: list = old_average
        new_average.append(average_value)
        new_average.pop(0)

        return new_average

    def make_Z_orders(self, order_depth, position, position_limit, buy):
        orders = []
        if buy == True:
            prices = [price for price in order_depth.sell_orders.keys()]
            if prices == []:
                return orders
            price = np.max(prices)
            buy_power = position_limit - position
            orders.append(Order(Product.SQUID_INK, int(price), buy_power))
        else:
            prices = [price for price in order_depth.buy_orders.keys()]
            if prices == []:
                return orders
            price = np.min(prices)
            sell_power = position_limit + position
            orders.append(Order(Product.SQUID_INK, int(price), -sell_power))
        return orders

    def close_position(self, order_depth, position):
        orders = []
        if position > 0:
            prices = [price for price in order_depth.buy_orders.keys()]
            if prices == []:
                return orders
            price = np.min(prices)
            orders.append(Order(Product.SQUID_INK, int(price), -position))
        elif position < 0:
            prices = [price for price in order_depth.sell_orders.keys()]
            if prices == []:
                return orders
            price = np.max(prices)
            orders.append(Order(Product.SQUID_INK, int(price), -int(position)))
        return orders

    def clean_np(self, obj):
        if isinstance(obj, dict):
            return {k: self.clean_np(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.clean_np(x) for x in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj


    def calc_ema(self, days, close, old_ema):
        return (close - old_ema)*(2 / (days + 1))+old_ema


    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        traderObject = {
            "KELP": [],
            "SQUID_INK": [],
            "SQUID_INK_RSI": [],
            "SQUID_INK_EMA": [],
            "SQUID_INK_EMA2": []
        }
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        result = {}

        if Product.SQUID_INK in self.params and Product.SQUID_INK in state.order_depths:
            squid_ink_position = (
                state.position[Product.SQUID_INK]
                if Product.SQUID_INK in state.position
                else 0
            )
            mid_price = self.market_mid_price(Product.SQUID_INK, state)
            delta_t = PARAMS[Product.SQUID_INK]["delta_t"]
            delta_rsi = PARAMS[Product.SQUID_INK]["delta_rsi"]
            delta_ema = PARAMS[Product.SQUID_INK]["delta_ema"]
            delta_ema2 = PARAMS[Product.SQUID_INK]["delta_ema2"]
            delta_max = np.max([delta_t, delta_rsi, delta_ema, delta_ema2])

            if len(traderObject["SQUID_INK"]) < delta_max:
                if mid_price == 0:
                    pass
                else:
                    traderObject["SQUID_INK"] = np.ones(delta_max+1) * mid_price
                    traderObject["SQUID_INK"] = traderObject["SQUID_INK"].tolist()
            elif mid_price == 0:
                mid_price = traderObject["SQUID_INK"][-1]


            mean_price = np.mean(traderObject["SQUID_INK"])
            std_price = np.std(traderObject["SQUID_INK"])
            rs = np.asarray(traderObject["SQUID_INK"][-delta_rsi:]) - np.asarray(traderObject["SQUID_INK"][-(delta_rsi+1):-1])
            avg_gain = np.mean([x for x in rs if x >= 0])
            avg_loss = np.mean([x for x in rs if x < 0])*-1

            if avg_gain == 0 and avg_loss == 0:
                rsi = 50
            elif math.isnan(avg_loss):
                rsi = 100

            elif math.isnan(avg_gain):
                rsi = 0
            else:
                diff = mid_price-traderObject["SQUID_INK"][-1]
                if diff > 0:
                    avg_gain = (avg_gain*13+diff)/14
                elif diff < 0:
                    avg_loss = (avg_loss*13+diff*-1)/14

                rsi = 100 - (100 / (1 + (avg_gain/avg_loss)))


            if len(traderObject["SQUID_INK_RSI"]) < 1:
                rsi_list = [rsi]
                traderObject["SQUID_INK_RSI"] = rsi_list

            old_rsi = traderObject["SQUID_INK_RSI"][-1]

            old_ema, old_ema2, ema, ema2 = 0,0,0,0

            if len(traderObject["SQUID_INK_EMA"]) < 1:
                if state.timestamp > delta_ema*100:
                    temp_ema = self.calc_ema(delta_ema, mid_price, np.mean(traderObject["SQUID_INK"][-delta_ema:]))
                    temp_ema_list = [temp_ema]
                    traderObject["SQUID_INK_EMA"] = temp_ema_list
                else: pass
            else:
                old_ema = traderObject["SQUID_INK_EMA"][-1]
                ema = self.calc_ema(delta_ema, mid_price, old_ema)


            if len(traderObject["SQUID_INK_EMA2"]) < 1:
                if state.timestamp > delta_ema2*100:
                    temp_ema2 = self.calc_ema(delta_ema2, mid_price, np.mean(traderObject["SQUID_INK"][-delta_ema2:]))
                    temp_ema2_list = [temp_ema2]
                    traderObject["SQUID_INK_EMA2"] = temp_ema2_list
                else: pass
            else:
                old_ema2 = traderObject["SQUID_INK_EMA2"][-1]
                ema2 = self.calc_ema(delta_ema2, mid_price, old_ema2)

            if std_price == 0:
                 Z = 0
            else:
                 Z = (mid_price - mean_price) / std_price


            if state.timestamp > delta_max*100:
                if math.ceil(old_ema2*2)/2 < math.ceil(old_ema*2)/2 and math.ceil(ema2*2)/2 > math.ceil(ema*2)/2 and rsi < 30:
                    squid_ink_orders = self.make_Z_orders(state.order_depths[Product.SQUID_INK], squid_ink_position,
                                                          LIMITS["SQUID_INK"], True)
                elif math.ceil(old_ema*2)/2 < math.ceil(old_ema2*2)/2 and math.ceil(ema*2)/2 > math.ceil(ema2*2)/2 and rsi > 70:
                    squid_ink_orders = self.make_Z_orders(state.order_depths[Product.SQUID_INK], squid_ink_position,
                                                          LIMITS["SQUID_INK"], False)
                elif math.ceil(old_ema * 2)/2 < mid_price <= math.ceil(ema * 2)/2:
                    squid_ink_orders = self.close_position(state.order_depths[Product.SQUID_INK], squid_ink_position)
                elif math.ceil(old_ema * 2)/2 > mid_price >= math.ceil(ema * 2)/2:
                    squid_ink_orders = self.close_position(state.order_depths[Product.SQUID_INK], squid_ink_position)
                else:
                    squid_ink_orders = []
                '''
                if std_price == 0:
                    Z = 0
                else:
                    Z = (mid_price - mean_price) / std_price

                #if Z > 2 and (old_rsi >= 70 > rsi):
                if Z > 2 and rsi < 70 <= old_rsi:
                    # Sell overpriced ink
                    squid_ink_orders = self.make_Z_orders(state.order_depths[Product.SQUID_INK], squid_ink_position, LIMITS["SQUID_INK"], False)
                #elif Z < -2 and (old_rsi <= 30 < rsi):
                elif Z < -2 and rsi > 30 >= old_rsi:
                    # Buy underpriced ink
                    squid_ink_orders = self.make_Z_orders(state.order_depths[Product.SQUID_INK], squid_ink_position, LIMITS["SQUID_INK"], True)
                elif np.abs(Z) > 7 or np.abs(rsi-50) < 5:
                    # Close position
                    squid_ink_orders = self.close_position(state.order_depths[Product.SQUID_INK], squid_ink_position)

                else:
                    squid_ink_orders = []
                '''

                result[Product.SQUID_INK] = squid_ink_orders
            traderObject["SQUID_INK"] = self.update_averager(mid_price, traderObject["SQUID_INK"])
            traderObject["SQUID_INK_RSI"] = self.update_averager(rsi, traderObject["SQUID_INK_RSI"])
            traderObject["SQUID_INK_EMA"] = self.update_averager(ema, traderObject["SQUID_INK_EMA"])
            traderObject["SQUID_INK_EMA2"] = self.update_averager(ema2, traderObject["SQUID_INK_EMA2"])
        # --------------------------return value ajustments----------------------------------------------------
        # this will become important later in the game
        conversions = 1

        traderObject = self.clean_np(traderObject)
        # backup the recent version of the logger object
        # trader_data_object["logger"] = logger
        # traderData = jsonpickle.encode(trader_data_object)

        logger.flush(state, result, conversions, traderObject)
        traderData = jsonpickle.encode(traderObject)



        return result, conversions, traderData