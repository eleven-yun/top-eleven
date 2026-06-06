# Glossary

Terms, abbreviations, and concepts used in this project.
Organized by topic for easier reading.

---

## Betting and Odds

### SP (Settlement Price / 派彩赔率)
The final payout odds at which a winning bet is paid out. In China Sports Lottery
(竞彩足球), SP is determined by the **parimutuel pool**: all money wagered on a
given outcome is pooled together, and after the house takes its cut (typically
25–30%), the remainder is divided among winners. This means SP is not set in
advance — it depends entirely on how bettors as a group distributed their money.

Compare to **EU bookmaker fixed odds**, where the payout is locked in when you
place the bet.

> Example: You bet ¥2 on Home Win at SP 2.10. If home wins, you receive
> ¥2 × 2.10 = ¥4.20 (profit ¥2.20). If another outcome wins, you lose ¥2.

### EU Odds (European Bookmaker Odds / 欧洲赔率)
Fixed odds set by bookmakers (e.g. Bet365, Pinnacle) before the match. Unlike CN
lottery SP, EU odds are quoted for all three outcomes (Home/Draw/Away) before
kick-off, making it possible to compute EV before betting. The house margin
(vig) on EU odds is typically 5–10%, much lower than CN lottery (~25–30%).

We use EU odds as a **proxy** for EV calculation when CN pre-match odds are not
available.

### EV (Expected Value / 期望值)
The average profit per ¥1 bet over many identical bets. Positive EV means the
strategy makes money in the long run.

$$\text{EV}_i = P(\text{outcome}_i) \times \text{odds}_i - 1$$

> Example: If the model says P(Home Win) = 0.55 and odds are 2.00:
> EV = 0.55 × 2.00 − 1 = +0.10 (+10%)
> Interpretation: on average, every ¥1 bet returns ¥1.10 over many trials.

A bet is placed only when EV exceeds a threshold (e.g. `--ev-threshold 0.05`).

### Vig / Juice / House Margin (抽水 / 佣金)
The percentage of every bet that the house keeps as profit, regardless of outcome.
Built into odds by making them sum to > 1.0 when converted to implied probabilities.

$$\text{Vig} = 1 - \frac{1}{\sum_i \frac{1}{\text{odds}_i}}$$

> Example: If Home=2.10, Draw=3.40, Away=3.20, then implied probs sum to
> 1/2.10 + 1/3.40 + 1/3.20 ≈ 1.09 → vig ≈ 9%.

CN lottery vig ≈ 25–30%. EU bookmakers ≈ 5–10%. Lower vig = more beatable.

### Parimutuel Betting (彩池式投注)
A betting system where all bets on an event are pooled together, the house takes
a fixed cut, and the remainder is shared among winners in proportion to their stake.

This is how China Sports Lottery works. The key implication: **you are betting
against other bettors, not the house**. If everyone bets on the favourite, the SP
for the favourite is low (bad value); contrarian picks on undervalued outcomes
pay more.

### Handicap / Asian Handicap (让球 / 亚盘)
A form of betting that gives a virtual goal advantage or disadvantage to one team
before the match starts, making both sides more balanced to bet on.

> Example: Handicap line = −1.0 (home gives 1 goal)
> If home wins 2-0, handicap result = Home Win (2−1=1 > 0)
> If home wins 1-0, handicap result = Draw (1−1=0)
> If home wins 1-0 with line −0.5, handicap result = Home Win (1−0.5=0.5 > 0)

In our data, `handicap_line` is from the home team's perspective. A negative value
means home team gives goals (handicap disadvantage); positive means home team
receives goals.

### 1X2 (胜平负)
Standard three-outcome betting market:
- **1** = Home Win (胜)
- **X** = Draw (平)
- **2** = Away Win (负)

The **fulltime_1x2** play type uses the final 90-minute result.
The **handicap_1x2** play type applies the handicap line first, then resolves 1X2.

### HT/FT (半场/全场, 半全场)
Halftime / Fulltime double outcome. Bets on the combination of the halftime result
AND the fulltime result (e.g. Home at halftime, Away at fulltime = H/A).
Nine possible outcomes: H/H, H/D, H/A, D/H, D/D, D/A, A/H, A/D, A/A.
High payout but very hard to predict.

### Closing Odds (盘口收盘赔率)
The final odds available immediately before kick-off, after the market has absorbed
all betting information. Closing odds are the most informative — professional
bettors move the market toward true probabilities throughout the week.

In our pipeline, we use closing odds as features (market signal) and for EV
calculation.

### ROI (Return on Investment / 投资回报率)
Total profit divided by total amount staked, expressed as a percentage.

$$\text{ROI} = \frac{\text{Total Profit}}{\text{Total Staked}} \times 100\%$$

> Example: 1000 bets × ¥2 stake = ¥2000 staked. Profit = ¥86.
> ROI = 86 / 2000 = +4.3%.

Positive ROI sustained over many bets indicates a genuine edge.

### Hit Rate (命中率)
Percentage of placed bets that won. A hit rate of 55.5% on a balanced 1X2 market
is meaningfully above the ~33% random baseline, but hit rate alone is not
sufficient — odds matter too (winning on low-odds favourites gives less value).

### Flat Stake (固定注额)
Betting the same fixed amount (e.g. ¥2) on every qualifying bet, regardless of
confidence or edge. Simple but suboptimal compared to Kelly sizing.

### Kelly Criterion (凯利公式)
An optimal bet-sizing formula that maximizes the long-run growth rate of your
bankroll. It stakes more on high-EV bets and less on marginal ones.

$$f^* = \frac{p \cdot b - (1-p)}{b} = \frac{p \cdot \text{odds} - 1}{\text{odds} - 1}$$

where $p$ = model probability, $b$ = net odds (odds − 1).

**Fractional Kelly** (e.g. half-Kelly, $f = f^*/2$) is commonly used to reduce
variance while retaining most of the growth advantage.

> Example: p=0.55, odds=2.00 → f* = (0.55×2 − 1) / (2−1) = 0.10 → stake 10% of bankroll.
> Half-Kelly: stake 5%.

### Drawdown (回撤)
The peak-to-trough decline in cumulative profit during a losing streak.
A max drawdown of −¥44 means at the worst point you were ¥44 below your
previous high-water mark. Important for bankroll management.

### Closing Line Value (CLV)
How well your bets beat the closing odds — a professional metric for edge quality.
If you consistently bet at odds better than the closing line, your model has genuine
predictive power. Not yet measured in this project.

---

## Model and ML

### LightGBM
"Light Gradient Boosting Machine" — a tree-based ML model (Microsoft, 2017).
Trains fast, handles tabular features well, and often outperforms neural networks on
structured data. Our primary model since Phase 5. Trained separately per task
(handicap, fulltime, HT/FT).

### Transformer
A neural network architecture based on self-attention (Vaswani et al. 2017,
"Attention is All You Need"). Originally designed for sequences (NLP). We adapted
it to encode a match-token sequence of pre-match features. Currently used as a
research path — LightGBM dominates it on all three tasks.

### Feature Engineering (特征工程)
The process of transforming raw data into informative numerical inputs for a model.
Examples in this project: rolling goal averages, ELO ratings, weighted form scores,
rest days. Good features matter more than model architecture for tabular data.

### Log Loss (对数损失)
The primary evaluation metric. Measures how well a model's predicted probabilities
match the true outcomes. Lower is better. A model predicting 50/50 on everything
would score around 1.099 on a 3-class problem (ln(3) ≈ 1.099).

$$\text{LogLoss} = -\frac{1}{N}\sum_i \log(p_i(\text{true class}))$$

### Brier Score
A quadratic scoring rule for probabilistic predictions. Like log loss, lower is
better. Less sensitive to extremely wrong predictions than log loss.

$$\text{Brier} = \frac{1}{N}\sum_i \sum_{k} (p_{ik} - y_{ik})^2$$

### ECE (Expected Calibration Error / 期望校准误差)
Measures whether predicted probabilities match empirical frequencies.
ECE = 0 means "when the model says 70% chance, it's right 70% of the time."
ECE < 0.015 is often considered well-calibrated for this project; the exact
value should be verified from the latest calibration report.

### Calibration (校准)
The process of adjusting raw model output probabilities so they align with real
outcome frequencies. Two common methods:
- **Temperature scaling**: divide logits by a scalar T > 1 to soften over-confident outputs
- **Isotonic regression**: fits a non-decreasing step function mapping raw scores to calibrated probs

### ELO Rating (ELO等级分)
A rating system (invented for chess by Arpad Elo) that assigns each team a
numeric strength score, updated after every match. A win against a strong opponent
gains more points than a win against a weak one. Used in this project as a
pre-match team strength feature.

### Ensemble (集成学习)
Combining predictions from multiple models to improve accuracy. We tested
soft-voting ensemble (weighted average of LightGBM and transformer probabilities).
Result: LightGBM alone dominated at every mixing ratio — transformer added noise.

### Overfitting (过拟合)
When a model learns patterns specific to training data that do not generalize to
unseen data. Detected by comparing validation ROI vs test ROI — a large gap
suggests overfitting to the validation season.

### Train / Validation / Test Split (训练/验证/测试集)
Season-based data split to simulate real forecasting:
- **Train**: seasons 1–N-2 (model learns from)
- **Validation**: season N-1 (used for hyperparameter tuning and early stopping)
- **Test**: season N (held out, touched once to report final performance)

Splitting by season (not randomly) is critical — random splits would leak future
match results into training data.

---

## China Lottery Specifics

### 竞彩足球 (Jingcai Football Lottery)
China's official football lottery product run by the Sports Lottery Administration
Center. Bettors pick outcomes on selected European and domestic league matches.
Uses parimutuel pools. Minimum bet is ¥2 per outcome per match.

### 期号 / Issue Number (e.g. 周六035)
Each lottery issue has a unique identifier: day of week + sequence number.
"周六035" = Saturday, issue 035. Used to group matches that are part of the same
betting slip.

### SPF (让球胜平负 / Handicap 1X2)
The handicap 1X2 play type in CN lottery. "让球" means "give away goals" (handicap).
Three outcomes resolved after applying the official handicap line.

### NSPF (胜平负 / Fulltime 1X2)
The fulltime 1X2 play type in CN lottery without handicap. "N" stands for "normal".

### 500.com
A major Chinese sports data and lottery platform (`500.com`). Provides:
- Post-match SP payouts at `zx.500.com/jczq/kaijiang.php?d=YYYY-MM-DD`
- Pre-match odds at `odds.500.com` (not yet scraped)

Pages use GB2312 encoding (not UTF-8).

---

## Project-Specific Terms

### EU Odds Proxy
Since we don't (yet) have pre-match CN lottery odds for all 3 outcomes, we use
European bookmaker closing odds (from football-data.co.uk) as a substitute for EV
calculation. The assumption is that EU market odds are an accurate reflection of
true outcome probabilities, and CN lottery odds (SP) will be proportional.

Phase 9 goal: replace the proxy with real pre-match CN odds.

### Match Coverage (匹配率)
Percentage of scraped CN lottery records that we successfully linked to a match in
our processed dataset via fuzzy matching. Currently ~46% — the rest are matches
from leagues not in our training data (UEFA CL, AFC CL, etc.).

### Issue Predict (期号预测)
Daily workflow that takes today's lottery issue schedule, runs the LightGBM model
on each match, applies EV and confidence filters, and outputs a ranked pick list.
Implemented in `scripts/issue_predict.py`.
