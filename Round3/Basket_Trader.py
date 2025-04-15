import json
import jsonpickle
import numpy as np
from typing import Any, List

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState


class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    BASKET1 = "PICNIC_BASKET1"
    BASKET2 = "PICNIC_BASKET2"
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBE = "DJEMBE"


LIMITS = {
    "RAINFOREST_RESIN": 50,
    "KELP": 50,
    "SQUID_INK": 50,
    "CROISSANTS": 250,
    "JAMS": 350,
    "DJEMBE": 60,
    "PICNIC_BASKET1": 60,
    "PICNIC_BASKET2": 100
}

PARAMS = {
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,
        "take_width": 1,
        "clear_width": 0.5,
        "volume_limit": 0,
    },
    Product.KELP: {
        "delta_t":  10,
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
    Product.BASKET2: {
        "delta_t": 2
    },
    Product.CROISSANTS: {

    },
    Product.JAMS: {

    },
    Product.BASKET1: {

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
            Product.KELP: 50
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

    def kelp_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price])
                   >= self.params[Product.KELP]["adverse_volume"]
            ]
            filtered_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price])
                   >= self.params[Product.KELP]["adverse_volume"]
            ]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            if mm_ask == None or mm_bid == None:
                if traderObject.get("kelp_last_price", None) == None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject["kelp_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2

            if traderObject.get("kelp_last_price", None) != None:
                last_price = traderObject["kelp_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = (
                        last_returns * self.params[Product.KELP]["reversion_beta"]
                )
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price
            traderObject["kelp_last_price"] = mmmid_price
            return fair
        return None

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

    def make_basket_orders(self, order_depth, position, position_limit, buy, mid_price):
        orders = []
        if buy == True:
            logger.print("BUY")
            prices = [price for price in order_depth.sell_orders.keys()]
            if prices == []:
                return orders
            buy_power = position_limit - position
            orders.append(Order(Product.BASKET2, int(mid_price), buy_power))
        else:
            logger.print("SELL")
            prices = [price for price in order_depth.buy_orders.keys()]
            if prices == []:
                return orders
            price = np.min(prices)
            sell_power = position_limit + position
            orders.append(Order(Product.BASKET2, int(mid_price), -sell_power))
        return orders

    def close_position(self, order_depth, position):
        orders = []
        if position > 0:
            prices = [price for price in order_depth.buy_orders.keys()]
            if prices == []:
                return orders
            price = np.min(prices)
            orders.append(Order(Product.BASKET2, int(price), -position))
        elif position < 0:
            prices = [price for price in order_depth.sell_orders.keys()]
            if prices == []:
                return orders
            price = np.max(prices)
            orders.append(Order(Product.BASKET2, int(price), -int(position)))
        return orders

    def hedge(self, product, order_depth, position, target, limit):
        orders = []
        if target > limit:
            target = limit
        elif target < -limit:
            target = -limit
        if target > position:
            prices = [price for price in order_depth.sell_orders.keys()]
            if prices == []:
                return orders
            buy_quantity = target - position
            price = np.max(prices)
            orders.append(Order(product, int(price), buy_quantity))
        elif position > target:
            prices = [price for price in order_depth.buy_orders.keys()]
            if prices == []:
                return orders
            sell_quantity = position - target
            price = np.min(prices)
            orders.append(Order(product, int(price), -sell_quantity))
        return orders

    def calc_averager(self, values: list) -> float:
        avg = 0
        for price in values:
            avg += price
        avg = avg / len(values)
        return avg

    def update_averager(self, average_value, old_average: list):
        new_average: list = old_average
        new_average.pop(0)
        new_average.append(average_value)

        return new_average


    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        traderObject = []
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        result = {}
        conversions = 0
        
        basket2_tradable = Product.BASKET2 in self.params and Product.BASKET2 in state.order_depths
        croissants_tradable = Product.CROISSANTS in self.params and Product.CROISSANTS in state.order_depths
        jams_tradable = Product.JAMS in self.params and Product.JAMS in state.order_depths

        logger.print(basket2_tradable, croissants_tradable, jams_tradable)

        if basket2_tradable and croissants_tradable and jams_tradable:
            basket2_position = (
                state.position[Product.BASKET2]
                if Product.BASKET2 in state.position
                else 0
            )
            croissants_position = (
                state.position[Product.CROISSANTS]
                if Product.CROISSANTS in state.position
                else 0
            )
            jams_position = (
                state.position[Product.JAMS]
                if Product.JAMS in state.position
                else 0
            )

            basket2_mid = self.market_price(Product.BASKET2, state)
            croissants_mid = self.market_price(Product.CROISSANTS, state)
            jams_mid = self.market_price(Product.JAMS, state)

            basket2_comp = 4 * croissants_mid + 2 * jams_mid + 36.5

            logger.print(basket2_mid - basket2_comp)

            if basket2_mid - basket2_comp < -60:
                basket2_orders = self.make_basket_orders(
                    state.order_depths[Product.BASKET2],
                    basket2_position,
                    LIMITS["PICNIC_BASKET2"],
                    True,
                    basket2_mid
                )

            elif basket2_mid - basket2_comp > 60:
                basket2_orders = self.make_basket_orders(
                    state.order_depths[Product.BASKET2],
                    basket2_position,
                    LIMITS["PICNIC_BASKET2"],
                    False,
                    basket2_mid
                )
            elif abs(basket2_mid - basket2_comp) < 40:
                basket2_orders = self.close_position(
                    state.order_depths[Product.BASKET2],
                    basket2_position
                )
            else:
                basket2_orders = []



            result[Product.BASKET2] = (
                basket2_orders
            )

            logger.print(result)

            # Hedge

            target_croissants = -4 * basket2_position
            target_jams = -2 * basket2_position

            croissants_orders = self.hedge(
                Product.CROISSANTS,
                state.order_depths[Product.CROISSANTS],
                croissants_position,
                target_croissants,
                LIMITS["CROISSANTS"],
            )

            jams_orders = self.hedge(
                Product.JAMS,
                state.order_depths[Product.JAMS],
                jams_position,
                target_jams,
                LIMITS["JAMS"],
            )

            result[Product.CROISSANTS] = (
                croissants_orders
            )

            result[Product.JAMS] = (
                jams_orders
            )


            # traderObject = self.update_averager(mid_price, traderObject)



        # --------------------------return value ajustments----------------------------------------------------
        # this will become important later in the game
        conversions = 1

        logger.flush(state, result, conversions, state.traderData)
        # backup the recent version of the logger object
        # trader_data_object["logger"] = logger
        # traderData = jsonpickle.encode(trader_data_object)

        traderData = jsonpickle.encode(traderObject)

        return result, conversions, traderData

