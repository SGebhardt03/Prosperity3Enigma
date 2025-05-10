# Overview 

This GitHub repository contains the code that we used during the IMC Prosperity 3 algorithmic trading challenge. The IMC Prosperity 3 Algorithmic Trading Challenge was an international, two-week online competition organized by IMC Trading. Participants work in teams and act as representatives of virtual islands in a simulated trading world. Their goal is to write algorithms that buy and sell fictional products (like resin, kelp, squid ink, etc.) to earn as much virtual currency ("SeaShells") as possible. The game runs in five rounds, each adding more complex financial products (like ETFs and options) and trading challenges. While the main focus is on algorithmic trading, there are also manual trading tasks that test quick thinking and decision-making. It’s designed to be educational and fun, giving students and early-career participants a hands-on feel for real-world trading strategies and markets. Below you can find our strategy for the individual products. 

<details>
  <summary>Rainforest Resin</summary>
  Rainforest resin is a product with a stable fair value of 10000 seashells. The price is stable and shifts sometimes around that value. We already implemented a strategy for such a product last year, but decided to use the strategy of the team <a href="https://github.com/ericcccsliu/imc-prosperity-2?tab=readme-ov-file">linear utility</a> from last year on amethysts because it performs slightly better than our algorithm. The difference is that their algorithms closes 0 ev positions, to be able to trade more often within the position limit.
  
</details>

<details>
  <summary>Kelp</summary>
  The price of kelp was drifting slowly, so we used the same strategy for kelp as for rainforest resin, but the fair value was given by an average over the last market prices. This strategy worked because the drift in the price was rather slow.
</details>

<details>
  <summary>Squid Ink</summary>
  This product was very challenging. The hint that was available at the prosperity wiki was: "Squid Ink can be a very volatile product with price having large swings. Making a two-sided market or carrying position can be risky for such an instrument. However, with large swings comes large reversion. Squid Ink prices show more tendency to revert short term swings in price.

A metric to keep track of the size of deviation/swing from recent average could help in trading profitable positions."

We tried to implement an algorithm using a z-score with an simple average and an exponential weighted average to detect price jumps and use the mean reverting property. We also tried to use a RSI metric to detect changes in the price, but everything we tried was not profitable. Our hypothesis is that there are price jumps in the "local" fair value where the price is reverting to. These jumps may eat up our profits. We ended up not trading squid ink at all.
</details>

<details>
  <summary>Baskets and their Components</summary>
  There were two baskets, basket 1 consists of 6 croissants, 3 jams and a djembe. Basket 2 consists of 4 croissants and 2 jams. Our first simple strategy was to compare the price of basket 2 to the price of its components, the difference had an average value of 30. So we used pair trading. After that we discovered that it is better to buy or sell the spread of basket 1 - 1.5 * basket 2 because the volatility of the difference to the price of djembe was higher. Therefore providing more profitable trading opportunities.


</details>

<details>
  <summary>Volcanic Rock Vouchers</summary>
  Tradable were volcanic rock and vouchers. Of course the vouchers modeled call options. We also got another hint for these products: 

  
  "Hello everyone, hope you're enjoying the VOLCANIC_ROCK vouchers and a variety of trading strategies these new products introduce. While digging for the rock, Archipelago residents found some ancient mathematics sharing insights into VOLCANIC_ROCK voucher trading. Here's what the message with obscure and advanced mathematics read,

Message begins,

I have discovered a strategy which will make ArchiCapital the biggest trading company ever. Here's how my thesis goes,

t: Timestamp
St: Voucher Underlying Price at t
K: Strike
TTE: Remaining Time till expiry at t
Vt: Voucher price of strike K at t

Compute,

m_t = log(K/St)/ sqrt(TTE)
v_t = BlackScholes ImpliedVol(St, Vt, K, TTE)

for each t, plot v_t vs m_t and fit a parabolic curve to filter random noise.

This fitted v_t(m_t) allows me to evaluate opportunities between different strikes. I also call fitted v_t(m_t=0) the base IV and I have identified interesting patterns in timeseries of base IV.

Message ends."


</details>


<details>
  <summary>Magnificent Macarons</summary>
  We did not implement a strategy for this product yet. Last year we already wrote a basic algorithm based on arbitrage.
</details>

# Result
