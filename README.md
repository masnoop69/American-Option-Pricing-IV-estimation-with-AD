# American Option Pricing using Binomial Tree, with Automatic Differentiation for IV Estimation (Proofs)

This proofs page is a walkthrough for pricing American options using the binomial model. It should be referred to alongside the notebook as it would be easier to understand. 

We use a classic binomial model to price American options, with the Cox-Ross-Rubinstein (CRR) method of parameter estimation (estimation of u and hence p using sigma). The binomial model is a simple, intuitive and effective way to price options, and it is a good starting point for understanding the mechanics of option pricing under risk-neutral, no-arbitrage assumptions. We do this through a vectorization approach, which is much faster than the standard approach of pricing each option separately and iteratively backwards.

In addition to that, based on the paper "Using the Newton-Raphson Method with Automatic Differentiation to Numerically Solve the Implied Volatility of Stock Options via the Binomial Model" by Michael Klibanov et al. [https://arxiv.org/html/2207.09033v3], we use the Newton-Raphson method and the automatic differentiation technique to solve for the implied volatility of the option in a way that is computationally faster than Brent's root search method (secant and bisection method), with high accuracy (RMSE from Brent's root search method is almost 0).

#### So what do the assumptions of risk-neutrality and no-arbitrage mean?

1. Risk-neutral world: We assume that investors do not expect a higher return for a higher perceived risk in options. This means that for any amount of incremental risk (probability of loss, etc) above a risk-free investment, the expected return is the same. Therefore, the expected return of an stock or option should be the same as the expected return of a risk-free investment (fixed deposit, money market rates, etc). This is a probability-focussed and forwards looking approach, and we look at our models as Martingale where discounted expectation of value of option in future equals to option value today.

2. Law of one price: 2 portfolios with the same payoff should have the same price. Hence, the price of a replicating portfolio (Stock - ZCB) should be the same as the price of the actual portfolio (Call option). Markets are assumed to be efficient.

#### Why is this so? How does this make sense?

In the BSM model and the binomial option pricing model, we look at the value of the option as a factor of the value of the underlying asset. This means that the underlying stock price already prices in investor-perceived risk. Therefore, as we are using formulas to calculate option value from the underlying asset, option value hence is only affected by underlying asset price, not investor risk preferences on the option itself.

## Introducing the Binomial Tree Model
<img width="1024" height="590" alt="image" src="https://github.com/user-attachments/assets/451d8b66-38af-4a3a-962c-a1e6fb7007e7" />

This is a simple model, where we let d = 1/u for convergence and control over the number of outcomes, under the CRR approach (which I will upload the proof at another time for derivation of u, d and p). Therefore, with u being the upstate multiplier and d being the downstate multiplier, the outcomes of the stock price model at terminal step N (and at any step n) can be modeled as:

$$
S_{N} = S_0 \cdot u^i \cdot d^j
$$

while the option payoffs at terminal step N:

$$
F_N = \begin{cases} 
\max(S_N - K, 0) & \text{for a Call} \\
\max(K - S_N, 0) & \text{for a Put}
\end{cases}
$$

The value of an American option would be the intrinsic value (exercise price at that time step) or the probability weighted average of the next option values of upstate and downstate, discounted back one time step. $\Delta t$ in this example case is therefore the time difference between any sequential steps.

At any step prior to maturity, the continuity value $F_{cont}$ = $F_{n}$ is the discounted expectation of the option's value $e^{-r \Delta t} \cdot \mathbb{E}^{\mathbb{Q}}[F_{n+1}]$ at the next time step ($n+1$).

$$
F_{cont} = e^{-r \Delta t} \cdot \left[ p \cdot F_{n+1}^{u} + (1 - p) \cdot F_{n+1}^{d} \right] = e^{-r \Delta t} \cdot \mathbb{E}^{\mathbb{Q}}[F_{n+1}]
$$

**Where:**
* $F_{cont}$: The value of holding the option for another period.
* $r$: Risk-free interest rate.
* $\Delta t$: Time step size ($T/N$).
* $p$: Risk-neutral probability of an "up" move.
* $F_{n+1}^{u}$: Option value at the next step in the "up" state.
* $F_{n+1}^{d}$: Option value at the next step in the "down" state.

And the intrinsic value or exercise value of the option at the current node (time $t = n \Delta t$):

$$
F_{intrinsic} = \begin{cases} 
\max(S_n - K, 0) & \text{for a Call} \\
\max(K - S_n, 0) & \text{for a Put}
\end{cases}
$$

**Where:**
* $S_n$: Stock price at the current node.
* $K$: Strike price.

Typically, American options are excercised where intrinsic value is greater than the continuation value of the option, hence $$F_n = \max \left( F_{cont}, \ F_{intrinsic} \right)$$. The payoff model at n would therefore be:

$$
F_n = \max \left( e^{-r \Delta t} \cdot \mathbb{E}^{\mathbb{Q}}[F_{n+1}], \ \begin{cases} S_n - K & \text{(Call)} \\ K - S_n & \text{(Put)} \end{cases} \right)
$$

By using an recursive method of backwards calculation for each and every step N to step 0, we can get the price of the option today. Keep in mind that for step n, there will always be n+1 outcomes.

## Vectorizing the Calculation

At the final time step $N$, we observe an array representing every possible stock price outcome. Instead of calculating each node individually, we use a single operation to create the (N+1) vector $S_N$:

$$\mathbf{S}_N = S_0 \cdot 
\begin{bmatrix} 
u^N d^0 \\ 
u^{N-1} d^1 \\ 
\vdots \\ 
u^0 d^N 
\end{bmatrix}$$

Hence, at maturity:

$$\mathbf{f}_N = \max(\mathbf{S}_N - K, 0)$$

To calculate the continuation value at time step $n$, we shift the values of the $(n+1)$ vector to align "up" and "down" movements. For any vector $\mathbf{f}_{n}$ of length $n+1$, we create two overlapping slices of length $n$:

$$
f_{i,\text{cont}} = e^{-r\Delta t}
\left[
p \cdot
\underbrace{
\begin{bmatrix}
f_0 \\
f_1 \\
\vdots \\
f_{M-1}
\end{bmatrix}
}_{\mathbf{f}[:-1]}
+
(1-p) \cdot
\underbrace{
\begin{bmatrix}
f_1 \\
f_2 \\
\vdots \\
f_M
\end{bmatrix}
}_{\mathbf{f}[1:]}
\right]
$$

**Where:**
* $f[:-1]$: All outcomes of option values in the up state at that step, vector of length n
* $f[1:]$: All outcomes of option values in the down state at that step, vector of length n

After which, we observe excercise conditions, and pass this vector through this filter:

$$\mathbf{f}_i = \max \left( \mathbf{f}_{i, \text{continuation}}, \ \mathbf{S}_i - K \right)$$

This process happens recursively backwards until time step 0, where there is the price of the option today.

## Automatic Differentiation and Vega 'Tree' Computation

At the same time of computation of option value at each step in the binomial tree, the paper proposes we calculate the vega simultaenously at each and every step, and this method is called Automatic Differentiation. Each intermediate step for each option value has its own Vega, and through a similar recursive method of calculation, option Vega at T = 0 can be calculated. This allows us to use the Newton-Ralphson's method for IV estimation, as Vega is the partial derivative of option price to its IV, and our objective function will take IV as its argument to return. For this computation reason, we assume that Vega at maturity is non-zero.

To calculate Vega ($\nu = \frac{\partial F}{\partial \sigma}$) efficiently within the binomial tree, we differentiate the pricing formula with respect to volatility $\sigma$ at each step. This allows us to propagate the sensitivity backwards alongside the option price.

#### 1. Terminal Sensitivity (At Step $N$)
At maturity, the stock price at node $i$ (where $i$ is the number of up steps) is given by the CRR parameters:

$$
S_{N,i} = S_0 \cdot u^i \cdot d^{N-i}
$$

Substituting $u = e^{\sigma\sqrt{\Delta t}}$ and $d = e^{-\sigma\sqrt{\Delta t}}$:

$$
S_{N,i} = S_0 \cdot e^{\sigma\sqrt{\Delta t} \cdot i} \cdot e^{-\sigma\sqrt{\Delta t} \cdot (N-i)} = S_0 \cdot e^{\sigma\sqrt{\Delta t} (2i - N)}
$$

Differentiating the stock price with respect to $\sigma$ gives the "Intrinsic Vega":

$$
\frac{\partial S_{N,i}}{\partial \sigma} = S_{N,i} \cdot \sqrt{\Delta t} \cdot (2i - N)
$$

The Vega of the option at maturity is therefore:

$$
\nu_N = \begin{cases} 
\frac{\partial S_{N,i}}{\partial \sigma} & \text{if Option is ITM} \\
0 & \text{if Option is OTM}
\end{cases}
$$

#### 2. Recursive Step (At Step $n < N$)
For any step prior to maturity, the option value $F_n$ is the discounted expectation of the future nodes:

$$
F_n = e^{-r\Delta t} \left[ p F_{n+1}^u + (1 - p) F_{n+1}^d \right]
$$

To find Vega ($\nu_n = \frac{\partial F_n}{\partial \sigma}$), we differentiate this equation using the **Product Rule**. Note that both the probability $p$ and the future option values $F_{n+1}$ depend on $\sigma$.

$$
\nu_n = e^{-r\Delta t} \left[ \underbrace{\frac{\partial p}{\partial \sigma} (F_{n+1}^u - F_{n+1}^d)}_{\text{Change in Probability}} + \underbrace{p \nu_{n+1}^u + (1 - p) \nu_{n+1}^d}_{\text{Expected Future Vega}} \right]
$$

#### 3. Sensitivity of Probability ($\frac{\partial p}{\partial \sigma}$)
The risk-neutral probability is defined as:

$$
p = \frac{e^{r\Delta t} - d}{u - d}
$$

Using the quotient rule $\left( \frac{f}{g} \right)' = \frac{f'g - fg'}{g^2}$, and knowing that $\frac{\partial u}{\partial \sigma} = u\sqrt{\Delta t}$ and $\frac{\partial d}{\partial \sigma} = -d\sqrt{\Delta t}$:

$$
\frac{\partial p}{\partial \sigma} = \frac{\sqrt{\Delta t} \cdot \left[ d(u - d) - (e^{r\Delta t} - d)(u + d) \right]}{(u - d)^2}
$$

## Newton-Ralphson Method

Newton-Ralphson Method is a root searching method through an iterative process. It follows as such.

### Introduction to the Newton-Raphson Method

The Newton-Raphson method is a powerful iterative technique used to find the roots of a real-valued function $f(x) = 0$. In the context of options pricing, we use it to reverse-engineer the **Implied Volatility (IV)** from the market price of an option.

#### 1. Geometric Interpretation
The method relies on linear approximation. Suppose we want to find the root $x$ where the curve $f(x)$ crosses the x-axis.

1.  **Initial Guess:** We start with an initial guess $x_1$.
2.  **Tangent Line:** We calculate the tangent line to the curve at $(x_1, f(x_1))$. The slope of this line is the derivative $f'(x_1)$.
3.  **Next Estimate:** We follow the tangent line down until it intersects the x-axis (where $y=0$). This intersection point becomes our next estimate, $x_2$.
4.  **Iteration:** We repeat this process until the difference between $x_n$ and $x_{n-1}$ is negligible (convergence).

#### 2. Mathematical Derivation
The slope of the tangent line at $x_{n-1}$ is given by the definition of the derivative:

$$
f'(x_{n-1}) = \frac{\Delta y}{\Delta x} = \frac{f(x_{n-1}) - 0}{x_{n-1} - x_n}
$$

Rearranging this equation to solve for the next approximation $x_n$:

$$
f'(x_{n-1}) (x_{n-1} - x_n) = f(x_{n-1})
$$

$$
x_{n-1} - x_n = \frac{f(x_{n-1})}{f'(x_{n-1})}
$$

$$
x_n = x_{n-1} - \frac{f(x_{n-1})}{f'(x_{n-1})}
$$

To find the Implied Volatility of an option, we define our objective function $F(\sigma)$ as the difference between our **Model Price** and the **Market Price**. We want to find the volatility $\sigma$ that makes this difference zero.

**Objective Function:**

$$
g(\sigma) = F_{model}(\sigma) - F_{market}
$$

Where:
* $F_{model}(\sigma)$ is the theoretical option price calculated using our binomial tree with volatility $\sigma$.
* $F_{market}$ is the observed market price (a constant).

**The Derivative (Vega):**
To use Newton-Raphson, we need the derivative of our objective function with respect to $\sigma$. Since $F_{market}$ is a constant, the derivative is simply the **Vega** ($\nu$) of the option, which we calculated via Automatic Differentiation.

$$
g'(\sigma) = \frac{\partial}{\partial \sigma} (F_{model}(\sigma) - F_{market}) = \frac{\partial F_{model}}{\partial \sigma} = \text{Vega}(\sigma)
$$

**The IV Update Formula:**
Substituting these into the general Newton-Raphson equation, we get the iterative update step for Implied Volatility:

$$
\sigma_n = \sigma_{n-1} - \frac{F_{model}(\sigma_{n-1}) - F_{market}}{\text{Vega}(\sigma_{n-1})}
$$

## Surface Plot and Evaluation

We check both methods to see if the plots are any different, but also to study how IV varies with T and log-moneyness. Visually, both plots look exactly the same. RMSE in this case is also very close to 0 (check code for this), and therefore it can be concluded that Newton's is better for IV computation, being much faster and with negligible loss (assuming Brent's is the absolute correct value and the benchmark for evaluation).

Brent's

<img width="900" height="800" alt="image" src="https://github.com/user-attachments/assets/7e8c92bb-c34b-4951-9fec-7b1481a170c8" />


Newton's

<img width="900" height="800" alt="image" src="https://github.com/user-attachments/assets/14767b8a-e268-4abf-944c-19738a4678a3" />


## Extension of Project: AD Computation of Greeks where Possible

### Calculation of Other Greeks

#### 1. Delta ($\Delta$): Recursive Expectation
Instead of a standard finite difference on the option value, we calculate Delta recursively by propagating the terminal Delta backwards through the tree.

**Terminal Step ($t=T$):**

$$
\Delta_N = \begin{cases} 
1 & \text{if } S_T > K \text{ (Call)} \\
-1 & \text{if } K > S_T \text{ (Put)} \\
0 & \text{otherwise}
\end{cases}
$$

**Recursive Step ($t < T$):**
We calculate the expected Delta at the current node based on the probability-weighted Deltas of the next step.

$$
\Delta_n = p \cdot \Delta_{n+1}^u + (1 - p) \cdot \Delta_{n+1}^d
$$

Where:

$$\Delta_{exercise} = \begin{cases} 
1 & \text{for a Call} \\
-1 & \text{for a Put}
\end{cases}$$

#### 2. Gamma ($\Gamma$): Finite Difference Method using Delta
Gamma measures the curvature of the option value (or the rate of change of Delta). We compute this using the finite difference of the Delta values calculated in the current layer, leading to n-1 array of Gamma in the n-1 layer, which is 1 less than the n number of Delta in the same layer. Therefore, a special handling is required for Gamma estimation in its own recursive loop, and this can be seen in the code.

$$
\Gamma_n = \frac{\Delta_n^u - \Delta_n^d}{S_n^u - S_n^d}
$$

Where:

$$\Gamma_{exercise} = 0$$

$$\Gamma_{N} = 0$$

#### 3. Rho ($\rho$): Automatic Differentiation
Rho measures sensitivity to the risk-free rate $r$. Since $r$ appears in both the discount factor ($e^{-r\Delta t}$) and the probability weights ($p$), we apply the product rule to the pricing formula:

$$
F = e^{-r\Delta t} [ p F_u + (1-p) F_d ]
$$

Differentiating with respect to $r$ yields three terms:
1.  **Discount Sensitivity:** $- \Delta t \cdot e^{-r\Delta t} [ p F_u + (1-p) F_d ]$ (The decay of the current price).
2.  **Probability Sensitivity:** $e^{-r\Delta t} \cdot \frac{\partial p}{\partial r} (F_u - F_d)$.
3.  **Recursive Rho:** $e^{-r\Delta t} [ p \rho_u + (1-p) \rho_d ]$.

Combining these gives the recursive update formula:

$$
\rho_n = \underbrace{e^{-r\Delta t} [ p \rho_{n+1}^u + (1-p) \rho_{n+1}^d ]}_{\text{Future Rho}} - \underbrace{\Delta t \cdot F_n}_{\text{Discount Decay}} + \underbrace{e^{-r\Delta t} \frac{\partial p}{\partial r} (F_{n+1}^u - F_{n+1}^d)}_{\text{Probability Sensitivity}}
$$

Where

$$\rho_{exercise} = 0$$

$$\rho_{N} = 0$$

#### 4. Theta ($\Theta$): Derived from PDE
Calculating Theta via finite difference on a tree can be noisy due to the discrete time steps. Instead, we solve for Theta algebraically using the **Black-Scholes-Merton Partial Differential Equation (PDE)**. An initial partial derivative solution was worked out by hand, but due to computational limits as $\Delta t$ was too small, hence 1/\sqrt{\Delta t} was too large and calculations were unstable.

The PDE states that for a risk-neutral portfolio:

$$
rF = \Theta + rS\Delta + \frac{1}{2}\sigma^2 S^2 \Gamma
$$

Rearranging for $\Theta$:

$$
\Theta = rF - rS\Delta - \frac{1}{2}\sigma^2 S^2 \Gamma
$$

This allows us to derive a highly stable Theta estimate using the $F$, $\Delta$, and $\Gamma$ values we have already computed at $t=0$.

### Comparison to the European BSM Model

Hence, for sanity check, we compare how the American Binomial Model greeks differ from the BSM Model greeks. Due to the early exercise condition in American Options, there would be differences.  

#### American Binomial Greeks

<img width="1961" height="1264" alt="image" src="https://github.com/user-attachments/assets/01a9a050-68b2-4000-b646-8fcdc266a7b9" />

#### BSM Greeks

<img width="1953" height="1262" alt="image" src="https://github.com/user-attachments/assets/66ef966c-c54a-470f-afee-7aa05cfebb7b" />

#### Binomial Greeks - BSM Greeks

<img width="1954" height="1261" alt="image" src="https://github.com/user-attachments/assets/4eb90482-6a9b-4e2e-b9d9-8337bb8d2401" />

Generally small differences other than Vega and Theta, which might arise from the continuous BSM vs the discrete Binomial model.



