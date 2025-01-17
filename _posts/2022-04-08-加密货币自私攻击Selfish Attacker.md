---
layout:     post
title:      比特币自私挖矿(Selfish Mining)
subtitle:   一种加密货币网络攻击方式
date:       2022-04-08
author:     Yinhao
header-img: img/blockchain_bg.jpg
catalog: true
usemathjax: true
tags:
    - Crypto Technology
---

# 1. 比特币自私挖矿(Selfish Mining)攻击方式
本文简述了比特币网络中Selfish Mining 自私挖矿的攻击方式。

## 1.1. 自私挖矿（Selfish Mining）攻击方式3步骤
1. 进入一个矿池
2. 偷偷地开辟一条分叉
3. 当挖到区块时，选择隐瞒当前区块而不是立即公布，然后继续下挖。目的就是为了领先公链。


我们把挖矿者分为两个阵营，1）诚实者和 2）自私者。
当诚实者抢先挖到新区块且广播后，自私者此时并没有攻击的选择。自私者会接着诚实者公布的新区块后继续挖掘。

## 1.2. Case 1 自私者挖到新区块就公布
![selfish 1](https://github.com/YinhaoDeng/yinhao.github.io/blob/master/img/Selfish_case1.png?raw=true)

*图中，蓝色方框代表公链区块头；粉色方框代表自私者挖到的区块，处于保密状态；绿色方框代表诚实者挖到的区块（默认为已公开状态）*

在状态（1）时，自私者率先发现一个新的区块，但保密。
在状态（2）时，诚实者也发现了一个新的区块，并立即公布于众。（默认操作不足为奇）
在状态（3）时，自私者选择公布其隐藏的区块。此时，存在2条长度相同的链。自私者肯定是继续在自己的链上挖掘，而诚实者可以二选其一继续挖掘。

接下来有3种可能：
**case 1**: 自私者在自己的链下抢先挖到下一个区块并且公布于众。此时自私者收益（revenue）为2，即挖出2个新区块所获得的奖励。

**case 2**: 诚实者在自私者的链头下继续开挖， 在自私者之前捷足先登并且公布于众。自私者收益到此为止，此时（revenue）为1，即挖出1个新区块所获得的奖励。

**case 3**: 诚实者在他们自己的链下挖到下一块区块且公布，自私者收益为0.即使自私者之前挖到一块，但是基于比特币网络中最长链（longest chain）准则，自私者挖到的链成为孤儿链（Orphan Chain）且作废。此时自私者收益（revenue）为0。白忙活一场。


## 1.3. Case 2 自私者选择隐瞒，不到迫不得已时才会公布
![selfish 2](https://github.com/YinhaoDeng/yinhao.github.io/blob/master/img/Selfish_case2.png?raw=true)
在状态（1）时，自私者率先发现一个新的区块，但保密。
在状态（2）时，自私者在其链下又发现了一个区块，继续保密。

接下来有2种可能：
**case 1**: 在状态（3）下，诚实者在公链下发现了一个新区块且立即公布，但是自私者领先一个区块。为了避免前文中”诚实者挖掘速度太快超越自己“的情况，自私者选择公布已经挖到的2个区块锁定收益，此时收益（revenue）为2.
*⚠️注意：不同于上文例子，诚实者不能在自私者链下挖掘，是因为自私者**隐瞒**了其挖掘到新区块的真相。在诚实者的视角下是**看不到**自私者这条链的。*

**case 2**: 自私者在自己的链上拼命挖，又挖到1个或好几个区块（不得不说运气挺好的），而诚实者原地踏步什么都没挖出来，如状态（5）。此时自私者已经领先其对手诚实者2个或2个以上的区块了，所以**自私者不慌，也不急着公布之前挖到的区块锁定收益**。


## 1.4. 一旦条件形成，自私者们就开始作祟了
![](https://github.com/YinhaoDeng/yinhao.github.io/blob/master/img/Selfish_attack.png?raw=true)

自私者们的具体操作方法如下：
1. 继续且只在自己的链上挖掘。(废话)
2. 努力保持至少2个区块的领先。若自私者被诚实者追击支只有1个区块领先时，自私者公布其挖的所有区块，锁定收益，套现走人。
3. 诚实者追上来一个区块，自私者就公布一个之前挖掘的区块。

如果自私者是少数（minority），即在所有矿工中占比 <50%， 那么有很大概率自私者领先的数量最终会降回1.


## 1.5. 关于自私挖矿攻击的研究
设：
系统里有矿工 $$1,...n$$, 矿工 $$i$$ 有$$m_{i}$$的算力，总算力为1：$$\sum_{i=1}^{n} m_{i}=1$$。自私者们的算力为$$\alpha$$，那么诚实者算力为$$（1-\alpha）$$。诚实者中选择在自私者矿池中挖掘的比例为$$\gamma$$，则诚实者中选择在诚实者矿池中挖掘的比例为$$（1-\gamma）$$。

我们先看一下正常情况下的收益率：
设自私矿池收益率为$$r_{pool}$$， 诚实者收益率为$$r_{others}$$。不难看出 $$r_{pool} \propto \alpha$$。理想情况下

$$
\begin{aligned}
  r_{pool} +r_{others}=1
\end{aligned}
$$

下面我们把自私者作祟的情况考虑进来：
自私者是**故意**要在网络中**创造多个分支**，这导致会有很多的区块没有被囊括进网络。所以区块生成速度比正常情况要慢。（区块没被承认，就没有奖励）

$$
\begin{aligned}
  r_{pool} +r_{others}<1
\end{aligned}
$$

每个矿工的真实收益率是收益率比率。

$$
\begin{aligned}
  R_{pool} = \frac{r_{pool}}{r_{pool}+r_{others}}= \frac{\alpha(1-\alpha^{2})(4\alpha+\gamma(1-2\alpha))-\alpha^{3}}{1-\alpha(1+(2-\alpha)\alpha)}
\end{aligned}
$$


假设诚实者占多数： 
$$0\le\alpha\le\frac{1}{2}$$
当自私者的算力满足以下要求时，自私者可以赢得更多的奖励： 
$$R_{pool}>\alpha$$
给定$$\gamma$$，当$$\alpha$$满足以下要求时，自私者矿池可以比应得的赢得更多：  

$$
\begin{aligned}
  \frac{1-\gamma}{3-2\gamma}\le\alpha\le\frac{1}{2}
\end{aligned}
$$


Reference:
英文原文链接：[Johns Hopkins Whiting School of Engineering](https://www.cs.jhu.edu/~abhishek/classes/CS601-641-441-Spring2018/Lecture8.pdf)
