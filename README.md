# American Option Pricing using Binomial Tree, with Automatic Differentiation for IV Estimation (Proofs)

This proofs page is a walkthrough for pricing American options using the binomial model. It should be referred to alongside the notebook as it would be easier to understand. 

We use a classic binomial model to price American options, with the Cox-Ross-Rubinstein (CRR) method of parameter estimation (estimation of u and hence p using sigma). The binomial model is a simple, intuitive and effective way to price options, and it is a good starting point for understanding the mechanics of option pricing under risk-neutral, no-arbitrage assumptions. We do this through a vectorization approach, which is much faster than the standard approach of pricing each option separately and iteratively backwards.

In addition to that, based on the paper "Using the Newton-Raphson Method with Automatic Differentiation to Numerically Solve the Implied Volatility of Stock Options via the Binomial Model" by Michael Klibanov et al. [https://arxiv.org/html/2207.09033v3], we use the Newton-Raphson method and the automatic differentiation technique to solve for the implied volatility of the option in a way that is computationally faster than Brent's root search method (secant and bisection method), with high accuracy (RMSE from Brent's root search method is almost 0).

#### So what do the assumptions of risk-neutrality and no-arbitrage mean?

1. Risk-neutral world: We assume that investors do not expect a higher return for a higher perceived risk in options. This means that for any amount of incremental risk (probability of loss, etc) above a risk-free investment, the expected return is the same. Therefore, the expected return of an stock or option should be the same as the expected return of a risk-free investment (fixed deposit, money market rates, etc).

2. Law of one price: 2 portfolios with the same payoff should have the same price. Hence, the price of a replicating portfolio (Stock - ZCB) should be the same as the price of the actual portfolio (Call option). Markets are assumed to be efficient.

#### Why is this so? How does this make sense?

In the BSM model and the binomial option pricing model, we look at the value of the option as a factor of the value of the underlying asset. This means that the underlying stock price already prices in investor-perceived risk. Therefore, as we are using formulas to calculate option value from the underlying asset, option value hence is only affected by underlying asset price, not investor risk preferences on the option itself.

## Introducing the Binomial Tree Model
<img width="600" height="500" alt="image" src="https://github.com/user-attachments/assets/45045ab0-fbcf-4f7c-b83a-c6ccfc6a35ab" />

This is a simple model, where we let d = 1/u for convergence and control over the number of outcomes, under the CRR approach (which I will upload the proof at another time for derivation of u, d and p). Therefore, with u being the upstate multiplier and d being the downstate multiplier, the outcomes of the model can be modeled as such.

Therefore, the payoff model is as such.

<img width="600" height="500" alt="image" src="https://github.com/user-attachments/assets/01cc5a1f-ac3f-4b82-a809-52059a3efd19" />

The value of an American option would be the intrinsic value (exercise price at that time step) or the probability weighted average of the next option values of upstate and downstate, discounted back one time step. Delta(t) in this example case is therefore the time difference between steps 2 and 3.
By using an recursive method of backwards calculation for each and every step N to step 0, we can get the price of the option today. Keep in mind that for step n, there will always be n+1 outcomes.

## Vectorizing the Calculation

<img width="600" height="500" alt="image" src="https://github.com/user-attachments/assets/edc2d9f0-1ba0-46b0-b9bb-94876f97dd8e" />

<img width="1180" height="500" alt="image" src="https://github.com/user-attachments/assets/f328f3eb-fb99-469d-926e-624864e1b77a" />

Hence, for each step, we take the 1st to n-1th outcomes to be multiplied by p and the 2nd to nth outcomes to be multiplied by (1-p), therefore condensing n outcomes to n-1 outcomes. This is a faster approach than the individual calculation of each outcome and doing the backwards method for each outcome.

## Automatic Differentiation and Vega 'Tree' Computation

At the same time of computation of option value at each step in the binomial tree, as also calculate the vega simultaenously at each and every step, and this method is called Automatic Differentiation. Each intermediate step for each option value has its own Vega, and through a similar recursive method of calculation, option Vega at T = 0 can be calculated. This allows us to use the Newton-Ralphson's method for IV estimation, as Vega is the partial derivative of option price to its IV, and our objective function will take IV as its argument to return.

We first calculate Vega at the Nth Step. When option value = 0 at N, vega must be equal to 0 as well.

<img width="600" height="500" alt="image" src="https://github.com/user-attachments/assets/867d8c40-324c-4438-99b8-4161d007db1e" />

i would therefore be represented as the number of upstates the underlying option at that particular price has, referring to the binomial tree in the introduction. Applying vectorization, i would therefore be an array where [N, N-1, N-2 ... ,0], and hence vega for every outcome at N can be computed, where there at N+1 outcomes.

Henceforth, the Vega at n, subsequently, is computed as such.

<img width="600" height="1200" alt="image" src="https://github.com/user-attachments/assets/885d62f0-c673-44a7-b147-6c9a0af100c0" />

In a similar fashion, cn+1(u) is represented by the vector outcomes of option values [C1, C2, ..., Cn-1], the first to the (n-1)th option outcome, and cn+1(d) being the vector outcomes [C2, C3, ..., CN]. The same will be done for vn+1(u) and vn+1(d).

## Newton-Ralphson Method

Newton-Ralphson Method is a root searching method through an iterative process. It follows as such.

<img width="600" height="1400" alt="image" src="https://github.com/user-attachments/assets/5281fc30-80f3-4d54-a4cf-fe061e576c77" />

Therefore, applying the same concepts to the pricing model to derive IV:

<img width="600" height="700" alt="image" src="https://github.com/user-attachments/assets/8931a5a3-29c4-48f9-8c4c-884b8dbbbcdb" />

## Surface Plot

Both plots look exactly the same. RMSE in this case is also very close to 0, and therefore it can be concluded that Newton's is better for IV computation, being faster and with negligible loss.

Brent's

<img width="600" height="500" alt="image" src="https://github.com/user-attachments/assets/ea5970c9-9b45-4d00-9b43-a3ebc9e090a3" />

Newton's

<img width="600" height="500" alt="image" src="https://github.com/user-attachments/assets/a3e5cb46-ac33-41df-82ba-3fbecfcec5b0" />




