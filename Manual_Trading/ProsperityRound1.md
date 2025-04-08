ProsperityRound1
================
2025-04-07

# Trade Objective

We’ve invited representatives from our neighboring islands to join us
for the celebratory Archipelago Trade-Off. You can trade Snowballs with
Pam the Penguin from the north, Pizza’s with Devin the Duck from the
south, and Silicon Nuggets with Benny the Bull from the west
archipelago.

Your objective is to trade these currencies and maximize your profit in
SeaShells. The number of trades is limited to 5. You must begin your
first trade and end your last trade with our own currency; SeaShells.
Use the trading table to develop your trading strategy, and use the drop
down fields to translate your strategy into actionable input. Once you
are satisfied with your strategy and input, use the ‘Submit manual
trade’ button to lock it in.

# Trading Table

``` r
currency.names <- c("Snowballs", "Pizza's", "SiliconNuggets", "SeaShells")
(trading.mat <- matrix(c(1,0.7,1.95,1.34,1.45,1,3.1,1.98,0.52,0.31,1,0.64,0.72,0.48,1.49,1), nrow = 4, ncol = 4, dimnames = list(currency.names,currency.names)))
```

    ##                Snowballs Pizza's SiliconNuggets SeaShells
    ## Snowballs           1.00    1.45           0.52      0.72
    ## Pizza's             0.70    1.00           0.31      0.48
    ## SiliconNuggets      1.95    3.10           1.00      1.49
    ## SeaShells           1.34    1.98           0.64      1.00

# Calculating

Number of trades is limited to 5, so all possible trades are listed
below (last trade has to be SeaShells).

``` r
possible.trades.mat <- as.matrix(expand.grid(1:4, 1:4, 1:4, 1:4,4))
head(possible.trades.mat)
```

    ##      Var1 Var2 Var3 Var4 Var5
    ## [1,]    1    1    1    1    4
    ## [2,]    2    1    1    1    4
    ## [3,]    3    1    1    1    4
    ## [4,]    4    1    1    1    4
    ## [5,]    1    2    1    1    4
    ## [6,]    2    2    1    1    4

Now we want to calculate the outcome based on a sequence.

``` r
calc_outcome <- function(seq.of.trades) {
  last <- 4 #first Trade SeaShells
  multiplier <- sapply(seq.of.trades, function(x) {
    mult <- trading.mat[last, x]
    assign('last', x, inherits = TRUE)
    return(mult)
  })
  prod(multiplier)
}
```

We apply the ´calc_outcome´ function to each sequence.

``` r
outcomes <- apply(possible.trades.mat, 1 , calc_outcome)
```

After getting the outcome for each sequence we search for the optimal
sequence.

``` r
optimal.seq <- currency.names[possible.trades.mat[which(outcomes == max(outcomes)),]]
paste0(c("SeaShells", optimal.seq), collapse = "->") #Starting with SeaShells
```

    ## [1] "SeaShells->Snowballs->SiliconNuggets->Pizza's->Snowballs->SeaShells"
