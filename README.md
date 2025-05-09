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
  This product was very challenging.
</details>

<details>
  <summary>Baskets and their Components</summary>
  
</details>

<details>
  <summary>Volcanic Rock Vouchers</summary>
  
</details>

</details>

<details>
  <summary>Magnificent Macarons</summary>
  
</details>

# Result
