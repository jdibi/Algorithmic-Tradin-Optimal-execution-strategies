# Algorithmic-Trading—Optimal Execution Strategies
Overview

This repository implements and explains Almgren–Chriss (AC) optimal execution: how to liquidate (or acquire) a large position over a fixed horizon while balancing market impact against price risk.

Goal: Minimize expected implementation shortfall plus a penalty for risk.

Key forces:

Temporary impact (η): You pay a worse price when you trade fast.

Permanent impact (γ): Your trades nudge the mid-price.

Volatility (σ): Waiting exposes you to price risk.

Risk aversion (λ): How much you dislike that risk.

The classic AC solution yields a closed-form trading schedule that front-loads more when risk aversion is high and flattens toward uniform when risk aversion is low.

Features

Discrete- and continuous-time AC model

Closed-form optimal trajectory 
𝑥(𝑡)x(t) and trading rate 𝑣(𝑡)v(t)
Linear temporary and permanent impact

Risk–return trade-off controlled by 
𝜆λ
Example notebooks for plotting trajectories and stress-testing parameters

Clean, dependency-light Python reference implementation

Mathematical model (short & sweet)

We want to sell $X_0 > 0$

	​

>0 shares over horizon 
[
0
,
𝑇
]
[0,T].
Let 
𝑥
(
𝑡
)
x(t) be shares remaining at time 
𝑡
t (so 
𝑥
(
0
)
=
𝑋
0
x(0)=X
0
	​

, 
𝑥
(
𝑇
)
=
0
x(T)=0), and 
𝑣
(
𝑡
)
=
−
𝑥
˙
(
𝑡
)
v(t)=−
x
˙
(t) be the sell rate.

Price dynamics (with permanent impact):

d
𝑆
𝑡
=
𝛾
 
𝑣
(
𝑡
)
 
d
𝑡
+
𝜎
 
d
𝑊
𝑡
dS
t
	​

=γv(t)dt+σdW
t
	​


Execution price (with temporary impact):

𝑆
~
𝑡
=
𝑆
𝑡
+
𝜂
 
𝑣
(
𝑡
)
S
~
t
	​

=S
t
	​

+ηv(t)

Objective (mean–variance of implementation shortfall):

min
⁡
𝑣
(
𝑡
)
𝐸
 ⁣
[
∫
0
𝑇
𝑣
(
𝑡
)
 
𝑆
~
𝑡
 
d
𝑡
]
  
+
  
𝜆
 
V
a
r
 ⁣
(
∫
0
𝑇
𝑣
(
𝑡
)
 
𝑆
𝑡
 
d
𝑡
)
v(t)
min
	​

E[∫
0
T
	​

v(t)
S
~
t
	​

dt]+λVar(∫
0
T
	​

v(t)S
t
	​

dt)

With linear impact and constant coefficients, the optimal continuous-time trajectory is:

𝑥
\*
(
𝑡
)
=
𝑋
0
 
sinh
⁡
 ⁣
(
𝜅
(
𝑇
−
𝑡
)
)
sinh
⁡
(
𝜅
𝑇
)
,
𝑣
\*
(
𝑡
)
=
𝜅
𝑋
0
 
cosh
⁡
 ⁣
(
𝜅
(
𝑇
−
𝑡
)
)
sinh
⁡
(
𝜅
𝑇
)
,
x
\*
(t)=X
0
	​

sinh(κT)
sinh(κ(T−t))
	​

,v
\*
(t)=κX
0
	​

sinh(κT)
cosh(κ(T−t))
	​

,

where

𝜅
=
𝜆
 
𝜎
2
𝜂
.
κ=
η
λσ
2
	​

	​

.

Low risk aversion (
𝜆
→
0
λ→0): 
𝑥
\*
(
𝑡
)
x
\*
(t) approaches a linear schedule (VWAP-like).

High risk aversion: front-loaded selling (finish sooner to cut risk).

𝛾
γ contributes to expected cost but doesn’t change the shape of 
𝑥
\*
(
𝑡
)
x
\*
(t) under the basic AC assumptions (it shifts the mean price path).

Discrete time (N slices, 
𝑡
𝑘
=
𝑘
Δ
𝑡
t
k
	​

=kΔt) has the analogous closed form:

𝑥
𝑘
=
𝑋
0
 
cosh
⁡
 ⁣
(
𝜅
Δ
𝑡
 
(
𝑁
−
𝑘
)
)
cosh
⁡
 ⁣
(
𝜅
Δ
𝑡
 
𝑁
)
,
𝑣
𝑘
=
𝑥
𝑘
−
1
−
𝑥
𝑘
.
x
k
	​

=X
0
	​

cosh(κΔtN)
cosh(κΔt(N−k))
	​

,v
k
	​

=x
k−1
	​

−x
k
	​

.
