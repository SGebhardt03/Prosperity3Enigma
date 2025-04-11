import json
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
        "delta_t": 15,
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

    def market_make(
            self,
            product: str,
            orders: List[Order],
            bid: int,
            ask: int,
            position: int,
            buy_order_volume: int,
            sell_order_volume: int,
    ) -> (int, int):
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, round(bid), buy_quantity))  # Buy order

        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, round(ask), -sell_quantity))  # Sell order
        return buy_order_volume, sell_order_volume

    def take_best_orders(
            self,
            product: str,
            fair_value: int,
            take_width: float,
            orders: List[Order],
            order_depth: OrderDepth,
            position: int,
            buy_order_volume: int,
            sell_order_volume: int,
            prevent_adverse: bool = False,
            adverse_volume: int = 0,
    ) -> (int, int):
        position_limit = self.LIMIT[product]
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]

            if best_ask <= fair_value - take_width:
                quantity = min(
                    best_ask_amount, position_limit - position
                )  # max amt to buy
                if quantity > 0:
                    orders.append(Order(product, best_ask, quantity))
                    buy_order_volume += quantity
                    order_depth.sell_orders[best_ask] += quantity
                    if order_depth.sell_orders[best_ask] == 0:
                        del order_depth.sell_orders[best_ask]

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if best_bid >= fair_value + take_width:
                quantity = min(
                    best_bid_amount, position_limit + position
                )  # should be the max we can sell
                if quantity > 0:
                    orders.append(Order(product, best_bid, -1 * quantity))
                    sell_order_volume += quantity
                    order_depth.buy_orders[best_bid] -= quantity
                    if order_depth.buy_orders[best_bid] == 0:
                        del order_depth.buy_orders[best_bid]
        return buy_order_volume, sell_order_volume

    def take_best_orders_with_adverse(
            self,
            product: str,
            fair_value: int,
            take_width: float,
            orders: List[Order],
            order_depth: OrderDepth,
            position: int,
            buy_order_volume: int,
            sell_order_volume: int,
            adverse_volume: int,
    ) -> (int, int):

        position_limit = self.LIMIT[product]
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]
            if abs(best_ask_amount) <= adverse_volume:
                if best_ask <= fair_value - take_width:
                    quantity = min(
                        best_ask_amount, position_limit - position
                    )  # max amt to buy
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity
                        order_depth.sell_orders[best_ask] += quantity
                        if order_depth.sell_orders[best_ask] == 0:
                            del order_depth.sell_orders[best_ask]

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if abs(best_bid_amount) <= adverse_volume:
                if best_bid >= fair_value + take_width:
                    quantity = min(
                        best_bid_amount, position_limit + position
                    )  # should be the max we can sell
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -1 * quantity))
                        sell_order_volume += quantity
                        order_depth.buy_orders[best_bid] -= quantity
                        if order_depth.buy_orders[best_bid] == 0:
                            del order_depth.buy_orders[best_bid]

        return buy_order_volume, sell_order_volume

    def make_rainforest_resin_orders(
            self,
            order_depth: OrderDepth,
            fair_value: int,
            position: int,
            buy_order_volume: int,
            sell_order_volume: int,
            volume_limit: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        baaf = min(
            [
                price
                for price in order_depth.sell_orders.keys()
                if price > fair_value + 1
            ]
        ) if not [
                     price
                     for price in order_depth.sell_orders.keys()
                     if price > fair_value + 1
                 ] == [] else 1000000
        bbbf = max(
            [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]
        ) if not [price for price in order_depth.buy_orders.keys() if price < fair_value - 1] == [] else 0

        if baaf <= fair_value + 2:
            if position <= volume_limit:
                baaf = fair_value + 3  # still want edge 2 if position is not a concern

        if bbbf >= fair_value - 2:
            if position >= -volume_limit:
                bbbf = fair_value - 3  # still want edge 2 if position is not a concern

        buy_order_volume, sell_order_volume = self.market_make(
            Product.RAINFOREST_RESIN,
            orders,
            bbbf + 1,
            baaf - 1,
            position,
            buy_order_volume,
            sell_order_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def make_kelp_orders(
            self,
            order_depth: OrderDepth,
            fair_value: int,
            position: int,
            buy_order_volume: int,
            sell_order_volume: int,
            volume_limit: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        baaf = min(
            [
                price
                for price in order_depth.sell_orders.keys()
                if price > fair_value + 1
            ]
        ) if not [
                     price
                     for price in order_depth.sell_orders.keys()
                     if price > fair_value + 1
                 ] == [] else 1000000
        bbbf = max(
            [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]
        ) if not [price for price in order_depth.buy_orders.keys() if price < fair_value - 1] == [] else 0

        if baaf <= fair_value + 2:
            if position <= volume_limit:
                baaf = fair_value + 3  # still want edge 2 if position is not a concern

        if bbbf >= fair_value - 2:
            if position >= -volume_limit:
                bbbf = fair_value - 3  # still want edge 2 if position is not a concern

        buy_order_volume, sell_order_volume = self.market_make(
            Product.KELP,
            orders,
            bbbf + 1,
            baaf - 1,
            position,
            buy_order_volume,
            sell_order_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def clear_position_order(
            self,
            product: str,
            fair_value: float,
            width: int,
            orders: List[Order],
            order_depth: OrderDepth,
            position: int,
            buy_order_volume: int,
            sell_order_volume: int,
    ) -> List[Order]:
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)

        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)

        if position_after_take > 0:
            # Aggregate volume from all buy orders with price greater than fair_for_ask
            clear_quantity = sum(
                volume
                for price, volume in order_depth.buy_orders.items()
                if price >= fair_for_ask
            )
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0:
            # Aggregate volume from all sell orders with price lower than fair_for_bid
            clear_quantity = sum(
                abs(volume)
                for price, volume in order_depth.sell_orders.items()
                if price <= fair_for_bid
            )
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume

    def take_orders(
            self,
            product: str,
            order_depth: OrderDepth,
            fair_value: float,
            take_width: float,
            position: int,
            prevent_adverse: bool = False,
            adverse_volume: int = 0,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        if prevent_adverse:
            buy_order_volume, sell_order_volume = self.take_best_orders_with_adverse(
                product,
                fair_value,
                take_width,
                orders,
                order_depth,
                position,
                buy_order_volume,
                sell_order_volume,
                adverse_volume,
            )
        else:
            buy_order_volume, sell_order_volume = self.take_best_orders(
                product,
                fair_value,
                take_width,
                orders,
                order_depth,
                position,
                buy_order_volume,
                sell_order_volume,
            )
        return orders, buy_order_volume, sell_order_volume

    def clear_orders(
            self,
            product: str,
            order_depth: OrderDepth,
            fair_value: float,
            clear_width: int,
            position: int,
            buy_order_volume: int,
            sell_order_volume: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume, sell_order_volume = self.clear_position_order(
            product,
            fair_value,
            clear_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
        )
        return orders, buy_order_volume, sell_order_volume



    def make_orders(
            self,
            product,
            order_depth: OrderDepth,
            fair_value: float,
            position: int,
            buy_order_volume: int,
            sell_order_volume: int,
            disregard_edge: float,  # disregard trades within this edge for pennying or joining
            join_edge: float,  # join trades within this edge
            default_edge: float,  # default edge to request if there are no levels to penny or join
            manage_position: bool = False,
            soft_position_limit: int = 0,
            # will penny all other levels with higher edge
    ):
        orders: List[Order] = []
        asks_above_fair = [
            price
            for price in order_depth.sell_orders.keys()
            if price > fair_value + disregard_edge
        ]
        bids_below_fair = [
            price
            for price in order_depth.buy_orders.keys()
            if price < fair_value - disregard_edge
        ]

        best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
        best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None

        ask = round(fair_value + default_edge)
        if best_ask_above_fair != None:
            if abs(best_ask_above_fair - fair_value) <= join_edge:
                ask = best_ask_above_fair  # join
            else:
                ask = best_ask_above_fair - 1  # penny

        bid = round(fair_value - default_edge)
        if best_bid_below_fair != None:
            if abs(fair_value - best_bid_below_fair) <= join_edge:
                bid = best_bid_below_fair
            else:
                bid = best_bid_below_fair + 1

        if manage_position:
            if position > soft_position_limit:
                ask -= 1
            elif position < -1 * soft_position_limit:
                bid += 1

        buy_order_volume, sell_order_volume = self.market_make(
            product,
            orders,
            bid,
            ask,
            position,
            buy_order_volume,
            sell_order_volume,
        )

        return orders, buy_order_volume, sell_order_volume

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

    def update_averager(self, average_value, old_average: list):
        new_average: list = old_average
        new_average.pop(0)
        new_average.append(average_value)

        return new_average

    def make_Z_orders(self, order_depth, position, position_limit, buy, mid_price):
        orders = []
        if buy == True:
            prices = [price for price in order_depth.sell_orders.keys()]
            if prices == []:
                return orders
            buy_power = position_limit - position
            orders.append(Order(Product.SQUID_INK, int(mid_price), buy_power))
        else:
            prices = [price for price in order_depth.buy_orders.keys()]
            if prices == []:
                return orders
            price = np.min(prices)
            sell_power = position_limit + position
            orders.append(Order(Product.SQUID_INK, int(mid_price), -sell_power))
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

    def is_monotonically_decreasing(self, arr):
        return np.all(np.diff(arr) <= 0)


    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        traderObject = {
            "KELP": [],
            "SQUID_INK_MEAN": 0,
            "SQUID_INK_VAR": 0
        }
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        result = {}
        conversions = 0

        if Product.RAINFOREST_RESIN in self.params and Product.RAINFOREST_RESIN in state.order_depths:
            rainforest_resin_position = (
                state.position[Product.RAINFOREST_RESIN]
                if Product.RAINFOREST_RESIN in state.position
                else 0
            )
            rainforest_resin_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.RAINFOREST_RESIN,
                    state.order_depths[Product.RAINFOREST_RESIN],
                    self.params[Product.RAINFOREST_RESIN]["fair_value"],
                    self.params[Product.RAINFOREST_RESIN]["take_width"],
                    rainforest_resin_position,
                )
            )
            rainforest_resin_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.RAINFOREST_RESIN,
                    state.order_depths[Product.RAINFOREST_RESIN],
                    self.params[Product.RAINFOREST_RESIN]["fair_value"],
                    self.params[Product.RAINFOREST_RESIN]["clear_width"],
                    rainforest_resin_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            rainforest_resin_make_orders, _, _ = self.make_rainforest_resin_orders(
                state.order_depths[Product.RAINFOREST_RESIN],
                self.params[Product.RAINFOREST_RESIN]["fair_value"],
                rainforest_resin_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.RAINFOREST_RESIN]["volume_limit"],
            )
            result[Product.RAINFOREST_RESIN] = (
                    rainforest_resin_take_orders + rainforest_resin_clear_orders + rainforest_resin_make_orders
            )

        if Product.KELP in self.params and Product.KELP in state.order_depths:
            kelp_position = (
                state.position[Product.KELP]
                if Product.KELP in state.position
                else 0
            )

            mid_price = self.market_price(Product.KELP, state)
            delta_t = PARAMS[Product.KELP]["delta_t"]

            if len(traderObject["KELP"]) < delta_t:
                if mid_price == 0:
                    pass
                else:
                    traderObject["KELP"] = np.ones(delta_t) * mid_price
                    traderObject["KELP"] = traderObject["KELP"].tolist()
            elif mid_price == 0:
                mid_price = traderObject["KELP"][-1]
            # Take the historic average price and do MM on that price
            fair_value = np.mean(traderObject["KELP"])

            kelp_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.KELP,
                    state.order_depths[Product.KELP],
                    fair_value,
                    self.params[Product.KELP]["take_width"],
                    kelp_position,
                )
            )

            kelp_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.KELP,
                    state.order_depths[Product.KELP],
                    fair_value,
                    self.params[Product.KELP]["clear_width"],
                    kelp_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )

            kelp_make_orders, _, _ = self.make_kelp_orders(
                state.order_depths[Product.KELP],
                fair_value,
                kelp_position,
                0,
                0,
                self.params[Product.KELP]["volume_limit"],
            )

            result[Product.KELP] = (
                kelp_make_orders
            )

            traderObject["KELP"] = self.update_averager(mid_price, traderObject["KELP"])

        if Product.SQUID_INK in self.params and Product.SQUID_INK in state.order_depths:
            squid_ink_position = (
                state.position[Product.SQUID_INK]
                if Product.SQUID_INK in state.position
                else 0
            )
            mid_price = self.market_price(Product.SQUID_INK, state)
            delta_t = PARAMS[Product.SQUID_INK]["delta_t"]

            # if len(traderObject["SQUID_INK_MEAN"]) < delta_t:
            #     if mid_price == 0:
            #         pass
            #     else:
            #         traderObject["SQUID_INK_MEAN"] = mid_price
            #         traderObject["SQUID_INK_VAR"] = 0
            # elif mid_price == 0:
            #     mid_price = traderObject["SQUID_INK_MEAN"]

            mean_price = np.mean(traderObject["SQUID_INK_MEAN"])
            std_price = np.sqrt(traderObject["SQUID_INK_VAR"])

            if std_price == 0:
                Z = 0
            else:
                Z = (mid_price - mean_price) / std_price

            logger.print("Z-Value:", Z)

            if Z < -1:
                # Sell overpriced ink
                squid_ink_orders = self.make_Z_orders(state.order_depths[Product.SQUID_INK], squid_ink_position, LIMITS["SQUID_INK"], False, mid_price)
            elif Z > 1:
                # Buy underpriced ink
                squid_ink_orders = self.make_Z_orders(state.order_depths[Product.SQUID_INK], squid_ink_position, LIMITS["SQUID_INK"], True, mid_price)
            elif np.abs(Z) < 0.1:
                # Close position
                logger.print(squid_ink_position)
                squid_ink_orders = self.close_position(state.order_depths[Product.SQUID_INK], squid_ink_position)
            elif np.abs(Z) > 2:
                logger.print(squid_ink_position)
                squid_ink_orders = self.close_position(state.order_depths[Product.SQUID_INK], squid_ink_position)
            else:
                squid_ink_orders = []

            result[Product.SQUID_INK] = squid_ink_orders

            traderObject["SQUID_INK_MEAN"] = 0.8 * traderObject["SQUID_INK_MEAN"] + 0.2 * mid_price
            traderObject["SQUID_INK_VAR"] = 0.8 * (mid_price - mean_price)**2 + 0.2 * traderObject["SQUID_INK_VAR"]

        # --------------------------return value ajustments----------------------------------------------------
        # this will become important later in the game
        conversions = 1

        # backup the recent version of the logger object
        # trader_data_object["logger"] = logger
        # traderData = jsonpickle.encode(trader_data_object)

        logger.flush(state, result, conversions, traderObject)
        traderData = jsonpickle.encode(traderObject)



        return result, conversions, traderData