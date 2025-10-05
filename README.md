# Algorithmic-Trading—Optimal Execution Strategies

## Overview

This repository implements and explains Almgren–Chriss (AC) optimal execution: how to liquidate (or acquire) a large position over a fixed horizon while balancing market impact against price movements risk.

Goal: Minimize expected implementation shortfall plus a penalty for risk(variance of the shortfall).

Key forces:

Temporary impact (η): You have a worse price when you trade fast.

Permanent impact (γ): Your trades nudge the mid-price.

Volatility (σ): Waiting exposes you to price risk.

Risk aversion (λ): How much you dislike that risk. ($\lambda > 0$ means you are risk averse)

linear market impact function leads the classic AC solution that yields a closed-form trading schedule that front-loads more when risk aversion is high and flattens toward uniform when risk aversion is low.

## Features:

Discrete- and continuous-time AC model

Closed-form optimal trajectory 𝑥(𝑡) and trading rate 𝑣(𝑡)

Linear temporary and permanent impact

Risk–return trade-off controlled by 𝜆

Example notebooks for plotting trajectories and stress-testing parameters

Clean, dependency-light Python reference implementation

## Mathematical model (short & sweet):

We want to sell $X_0 >0$ shares ove the horizon $[0, T]$.
Let $x(t)$ be shares remaining at time $t$
$x(0) = X$, $x(t) = 0$, $v(t) = -\dot{x}_t$ be the sell rate.

Price dynamics (with permanent impact):

$$dS_t =  - \gamma v(t)dt + \sigma dW_t$$

Execution price (with temporary impact): 

$$\tilde{S}_t = S_t - \eta v(t)$$

Objective (mean–variance of implementation shortfall):

$$\min_{v(t)} E[\int_0^T v(t)\tilde{S}_t dt] + \lambda Var(\int_0^T v(t) S_t dt)$$

the optimal continuous-time trajectory is:

$$x^*(t) =X\frac{sinh(\kappa (T-t))}{sinh(\kappa T)} $$

$$v^*(t) = \kappa X \frac{cosh(\kappa (T-t))}{sinh(\kappa T)}$$

where:

$$\kappa = \sqrt{\frac{\lambda \sigma^2}{\eta}}$$

Low risk aversion ($\lambda \rightarrow 0$): $x_t$*$ approaches a linear schedule (VWAP-like).

High risk aversion: front-loaded selling (finish sooner to cut risk).

𝛾 contributes to expected cost but doesn’t change the shape of  under the basic AC assumptions (it shifts the mean price path).

Discrete time (N slices, t_k = k\Deta t) has the analogous closed form:

$$x_k =  X\frac{sinh(\kappa (T-t_k))}{sinh(\kappa T)}$$

$$v_k = (1/\Delta t) (x_{k - 1} - x_k)$$

