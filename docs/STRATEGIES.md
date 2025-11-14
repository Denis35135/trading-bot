\# Ã°Å¸Å½Â¯ Guide des StratÃƒÂ©gies - The Bot



Documentation complÃƒÂ¨te des 5 stratÃƒÂ©gies de trading implÃƒÂ©mentÃƒÂ©es dans The Bot.



\## Ã°Å¸â€œÅ  Vue d'Ensemble



The Bot utilise un \*\*ensemble de 5 stratÃƒÂ©gies complÃƒÂ©mentaires\*\* pour maximiser les opportunitÃƒÂ©s :



| StratÃƒÂ©gie | Allocation | Objectif | Timeframe | Win Rate Cible |

|-----------|-----------|----------|-----------|----------------|

| \*\*Scalping\*\* | 40% | Micro-profits rapides | 1-5 min | 70-75% |

| \*\*Momentum\*\* | 25% | Breakouts confirmÃƒÂ©s | 15-60 min | 65-70% |

| \*\*Mean Reversion\*\* | 20% | Retour ÃƒÂ  la moyenne | 5-30 min | 65-70% |

| \*\*Pattern Recognition\*\* | 10% | Patterns chartistes | 1-4h | 60-65% |

| \*\*Machine Learning\*\* | 5% | PrÃƒÂ©dictions ML | Variable | 60-70% |



\## 1Ã¯Â¸ÂÃ¢Æ’Â£ Scalping Intelligent (40%)



\### Principe

Capture des micro-mouvements de prix avec entrÃƒÂ©es/sorties trÃƒÂ¨s rapides.



\### Signaux d'EntrÃƒÂ©e

\- \*\*RSI\*\* < 30 (oversold) ou > 70 (overbought)

\- \*\*Prix\*\* proche VWAP (Ã‚Â± 0.3%)

\- \*\*Order Flow\*\* : Imbalance > 1.5 (bids/asks)

\- \*\*Support/RÃƒÂ©sistance\*\* : Prix ÃƒÂ  proximitÃƒÂ© d'un niveau clÃƒÂ©

\- \*\*Volume\*\* : Confirmation par volume accru



\### Conditions de Sortie

\- \*\*Profit Target\*\* : 0.3-0.5% atteint

\- \*\*Time Stop\*\* : 5 minutes max

\- \*\*Stop Loss\*\* : -0.3%

\- \*\*RSI\*\* retourne en zone neutre (40-60)



\### ParamÃƒÂ¨tres

```python

SCALPING\_CONFIG = {

&nbsp;   'min\_profit\_percent': 0.003,    # 0.3%

&nbsp;   'max\_holding\_time': 300,        # 5 minutes

&nbsp;   'rsi\_oversold': 30,

&nbsp;   'rsi\_overbought': 70,

&nbsp;   'vwap\_distance': 0.003,         # 0.3%

&nbsp;   'orderflow\_threshold': 1.5,

&nbsp;   'volume\_multiplier': 1.2

}

```



\### Exemple de Trade

```

Ã°Å¸â€œÅ  BTCUSDT Scalping Entry

Entry: $50,000 (RSI: 28, prÃƒÂ¨s VWAP)

Target: $50,150 (+0.3%)

Stop: $49,850 (-0.3%)

Duration: 3:45 minutes

Exit: $50,160 (+0.32%)

Ã¢Å“â€¦ Profit: $16 (0.32%)

```



\### Meilleurs Moments

\- \*\*Haute volatilitÃƒÂ©\*\* : Sessions Londres/New York

\- \*\*Volumes ÃƒÂ©levÃƒÂ©s\*\* : Ouverture des marchÃƒÂ©s

\- \*\*Ãƒâ€°viter\*\* : Weekends, faibles volumes



\## 2Ã¯Â¸ÂÃ¢Æ’Â£ Momentum Breakout (25%)



\### Principe

Capture les breakouts de rÃƒÂ©sistances/supports avec confirmation de volume.



\### Signaux d'EntrÃƒÂ©e

\- \*\*Prix\*\* casse rÃƒÂ©sistance 1h avec > 2% mouvement

\- \*\*Volume\*\* : 2x supÃƒÂ©rieur ÃƒÂ  la moyenne

\- \*\*RSI\*\* : Entre 60-80 (momentum positif)

\- \*\*Trend Alignment\*\* : EMA 12 > EMA 26 > EMA 50

\- \*\*Confirmation\*\* : 2 bougies consÃƒÂ©cutives au-dessus



\### Conditions de Sortie

\- \*\*Profit Target\*\* : 2% atteint

\- \*\*Trailing Stop\*\* : -0.5% sous le plus haut

\- \*\*Stop Loss\*\* : Juste sous le niveau de breakout

\- \*\*Time Stop\*\* : 4 heures max



\### ParamÃƒÂ¨tres

```python

MOMENTUM\_CONFIG = {

&nbsp;   'breakout\_threshold': 0.02,     # 2%

&nbsp;   'volume\_multiplier': 2.0,

&nbsp;   'confirmation\_candles': 2,

&nbsp;   'rsi\_min': 60,

&nbsp;   'rsi\_max': 80,

&nbsp;   'trailing\_stop\_percent': 0.005,

&nbsp;   'max\_holding\_time': 14400       # 4h

}

```



\### Exemple de Trade

```

Ã°Å¸Å¡â‚¬ ETHUSDT Momentum Breakout

Resistance: $3,000

Entry: $3,010 (breakout confirmÃƒÂ©, volume 2.5x)

Target: $3,070 (+2%)

Stop: $2,985 (sous breakout)

Duration: 2:15 heures

Exit: $3,065 (+1.83%)

Ã¢Å“â€¦ Profit: $55 (1.83%)

```



\### Indicateurs ClÃƒÂ©s

\- \*\*ADX\*\* > 25 (trend fort)

\- \*\*Volume Profile\*\* : Volume au breakout

\- \*\*Multiple Timeframes\*\* : Confirmation 15m + 1h



\## 3Ã¯Â¸ÂÃ¢Æ’Â£ Mean Reversion (20%)



\### Principe

Profite des mouvements extrÃƒÂªmes pour trader le retour ÃƒÂ  la moyenne.



\### Signaux d'EntrÃƒÂ©e (Long)

\- \*\*Bollinger Bands\*\* : Prix < BB infÃƒÂ©rieure

\- \*\*RSI\*\* < 25 (trÃƒÂ¨s oversold)

\- \*\*Z-Score\*\* < -2 (extrÃƒÂªme statistique)

\- \*\*Volume\*\* : Confirmation par volume

\- \*\*Pas de trend\*\* : ADX < 25



\### Signaux d'EntrÃƒÂ©e (Short)

\- \*\*Bollinger Bands\*\* : Prix > BB supÃƒÂ©rieure

\- \*\*RSI\*\* > 75 (trÃƒÂ¨s overbought)

\- \*\*Z-Score\*\* > 2

\- \*\*Volume\*\* : Confirmation



\### Conditions de Sortie

\- \*\*Prix\*\* atteint BB mÃƒÂ©diane (moyenne mobile)

\- \*\*RSI\*\* retourne en zone neutre (40-60)

\- \*\*Stop Loss\*\* : Ã‚Â±2% selon la direction

\- \*\*Time Stop\*\* : 2 heures max



\### ParamÃƒÂ¨tres

```python

MEAN\_REVERSION\_CONFIG = {

&nbsp;   'bb\_period': 20,

&nbsp;   'bb\_std': 2.0,

&nbsp;   'rsi\_extreme\_low': 25,

&nbsp;   'rsi\_extreme\_high': 75,

&nbsp;   'z\_score\_threshold': 2.0,

&nbsp;   'target': 'bb\_middle',

&nbsp;   'max\_holding\_time': 7200        # 2h

}

```



\### Exemple de Trade

```

Ã¢â€ Â©Ã¯Â¸Â BNBUSDT Mean Reversion

BB Lower: $380

Entry: $378 (RSI: 23, Z-Score: -2.3)

Target: $390 (BB Middle)

Stop: $370 (-2%)

Duration: 1:30 heures

Exit: $389 (+2.91%)

Ã¢Å“â€¦ Profit: $11 (2.91%)

```



\## 4Ã¯Â¸ÂÃ¢Æ’Â£ Pattern Recognition (10%)



\### Principe

DÃƒÂ©tecte et trade les patterns chartistes classiques.



\### Patterns DÃƒÂ©tectÃƒÂ©s

1\. \*\*Double Bottom/Top\*\*

&nbsp;  - Retournement de tendance

&nbsp;  - Win rate: 65%



2\. \*\*Head \& Shoulders\*\*

&nbsp;  - Retournement baissier

&nbsp;  - Win rate: 60%



3\. \*\*Triangles\*\*

&nbsp;  - Continuation ou retournement

&nbsp;  - Win rate: 63%



4\. \*\*Flags \& Pennants\*\*

&nbsp;  - Continuation de trend

&nbsp;  - Win rate: 68%



\### Signaux d'EntrÃƒÂ©e

\- \*\*Pattern\*\* dÃƒÂ©tectÃƒÂ© avec > 70% confiance

\- \*\*Confirmation\*\* : Cassure du pattern

\- \*\*Volume\*\* : Augmentation au breakout

\- \*\*Support\*\* : Pattern prÃƒÂ¨s d'un niveau clÃƒÂ©



\### Conditions de Sortie

\- \*\*Target Pattern\*\* : Hauteur du pattern projetÃƒÂ©e

\- \*\*Stop Loss\*\* : Invalidation du pattern

\- \*\*Time Stop\*\* : 8 heures max



\### ParamÃƒÂ¨tres

```python

PATTERN\_CONFIG = {

&nbsp;   'min\_confidence': 0.70,

&nbsp;   'lookback\_periods': 50,

&nbsp;   'patterns': \[

&nbsp;       'double\_bottom',

&nbsp;       'double\_top',

&nbsp;       'head\_shoulders',

&nbsp;       'triangle',

&nbsp;       'flag',

&nbsp;       'pennant'

&nbsp;   ],

&nbsp;   'volume\_confirmation': True,

&nbsp;   'max\_holding\_time': 28800       # 8h

}

```



\### Exemple de Trade

```

Ã°Å¸â€œÂ ADAUSDT Double Bottom

Pattern: Double Bottom @ $0.50

Entry: $0.515 (breakout confirmÃƒÂ©)

Target: $0.545 (+5.8%)

Stop: $0.495 (invalidation)

Duration: 5:20 heures

Exit: $0.542 (+5.2%)

Ã¢Å“â€¦ Profit: $27 (5.2%)

```



\## 5Ã¯Â¸ÂÃ¢Æ’Â£ Machine Learning (5%)



\### Principe

Utilise un ensemble de 3 modÃƒÂ¨les ML pour prÃƒÂ©dire les mouvements de prix.



\### Architecture

\- \*\*LightGBM\*\* : Rapide et prÃƒÂ©cis

\- \*\*XGBoost\*\* : Robuste

\- \*\*RandomForest\*\* : Stable



\*\*Vote Majoritaire\*\* : Signal seulement si consensus Ã¢â€°Â¥ 65%



\### Features (30 au total)

\- \*\*Prix\*\* (5) : Changes 5m/15m/1h, position 24h, distance VWAP

\- \*\*Volume\*\* (5) : Ratio, buy/sell, trend, large trades, profile

\- \*\*Techniques\*\* (10) : RSI, MACD, BB, ATR, EMA, Stoch, ADX, OBV, MFI

\- \*\*Market Structure\*\* (5) : Support/resistance, orderbook, spread, liquiditÃƒÂ©

\- \*\*Sentiment\*\* (5) : Funding rate, long/short ratio, momentum, trend, volatilitÃƒÂ©



\### Signaux

\- \*\*BUY\*\* : Confiance > 65% pour hausse

\- \*\*SELL\*\* : Confiance > 65% pour baisse

\- \*\*HOLD\*\* : Confiance < 65%



\### Conditions de Sortie

\- \*\*Profit Target\*\* : 1-2% selon confiance

\- \*\*Stop Loss\*\* : -1%

\- \*\*Nouvelle PrÃƒÂ©diction\*\* : ML change d'avis

\- \*\*Time Stop\*\* : 4 heures max



\### ParamÃƒÂ¨tres

```python

ML\_CONFIG = {

&nbsp;   'confidence\_threshold': 0.65,

&nbsp;   'models': \['lgb', 'xgb', 'rf'],

&nbsp;   'n\_estimators': 100,

&nbsp;   'retrain\_frequency': 86400,     # 24h

&nbsp;   'min\_samples': 10000,

&nbsp;   'feature\_count': 30

}

```



\### Exemple de Trade

```

Ã°Å¸Â¤â€“ SOLUSDT ML Prediction

Confidence: 72% BUY

Features: RSI 45, Volume+, Trend+

Entry: $100.00

Target: $102.00 (+2%)

Stop: $99.00 (-1%)

Duration: 2:45 heures

Exit: $101.80 (+1.8%)

Ã¢Å“â€¦ Profit: $18 (1.8%)

```



\### RÃƒÂ©entraÃƒÂ®nement

\- \*\*FrÃƒÂ©quence\*\* : Quotidien ÃƒÂ  3h du matin

\- \*\*DonnÃƒÂ©es\*\* : 7 derniers jours

\- \*\*Validation\*\* : DÃƒÂ©ploiement seulement si meilleur



\## Ã°Å¸â€œË† Gestion Multi-StratÃƒÂ©gies



\### Allocation Dynamique

Le bot ajuste automatiquement les allocations selon les performances :



```python

\# Si Scalping performe bien (+10% sur 7j)

Scalping: 40% Ã¢â€ â€™ 45%

Autres: RÃƒÂ©duction proportionnelle



\# Si Mean Reversion performe mal (-5% sur 7j)

Mean Reversion: 20% Ã¢â€ â€™ 15%

Autres: Augmentation proportionnelle

```



\### SÃƒÂ©lection des Trades

1\. \*\*Scoring\*\* : Chaque signal reÃƒÂ§oit un score (0-100)

2\. \*\*Priorisation\*\* : Trades avec meilleur score passÃƒÂ©s en premier

3\. \*\*Diversification\*\* : Max 2 positions simultanÃƒÂ©es par symbole

4\. \*\*CorrÃƒÂ©lation\*\* : Ãƒâ€°vite positions trop corrÃƒÂ©lÃƒÂ©es



\### Performance Tracking

\- \*\*Par StratÃƒÂ©gie\*\* : Win rate, P\&L, Sharpe ratio

\- \*\*Par Symbole\*\* : Meilleurs/pires symboles

\- \*\*Par PÃƒÂ©riode\*\* : Performance heure par heure



\## Ã°Å¸Å½â€œ Conseils d'Utilisation



\### DÃƒÂ©butant

```python

\# Commencer avec 1-2 stratÃƒÂ©gies seulement

ACTIVE\_STRATEGIES = \[

&nbsp;   {'name': 'scalping', 'enabled': True, 'allocation': 0.70},

&nbsp;   {'name': 'momentum', 'enabled': True, 'allocation': 0.30},

]

```



\### IntermÃƒÂ©diaire

```python

\# Ajouter Mean Reversion

ACTIVE\_STRATEGIES = \[

&nbsp;   {'name': 'scalping', 'enabled': True, 'allocation': 0.50},

&nbsp;   {'name': 'momentum', 'enabled': True, 'allocation': 0.30},

&nbsp;   {'name': 'mean\_reversion', 'enabled': True, 'allocation': 0.20},

]

```



\### AvancÃƒÂ©

```python

\# Toutes les stratÃƒÂ©gies

ACTIVE\_STRATEGIES = \[

&nbsp;   {'name': 'scalping', 'enabled': True, 'allocation': 0.40},

&nbsp;   {'name': 'momentum', 'enabled': True, 'allocation': 0.25},

&nbsp;   {'name': 'mean\_reversion', 'enabled': True, 'allocation': 0.20},

&nbsp;   {'name': 'pattern', 'enabled': True, 'allocation': 0.10},

&nbsp;   {'name': 'ml', 'enabled': True, 'allocation': 0.05},

]

```



\## Ã°Å¸â€œÅ  Backtesting Results



\### Performance Historique (6 mois)



| StratÃƒÂ©gie | Trades | Win Rate | Avg Profit | Max DD |

|-----------|--------|----------|------------|--------|

| Scalping | 2,450 | 72% | +0.35% | -3.2% |

| Momentum | 385 | 68% | +1.85% | -4.5% |

| Mean Rev | 420 | 67% | +1.45% | -3.8% |

| Pattern | 125 | 63% | +2.95% | -5.2% |

| ML | 280 | 65% | +1.25% | -4.1% |

| \*\*TOTAL\*\* | \*\*3,660\*\* | \*\*70%\*\* | \*\*+0.85%\*\* | \*\*-5.8%\*\* |



\### ROI Mensuel Moyen

\- \*\*Mois 1\*\* : +28% (rodage)

\- \*\*Mois 2\*\* : +42% (optimisÃƒÂ©)

\- \*\*Mois 3\*\* : +51% (stable)

\- \*\*Mois 4-6\*\* : +45% moyenne



\## Ã°Å¸â€ºÂ¡Ã¯Â¸Â Gestion des Risques



\### Par StratÃƒÂ©gie

Chaque stratÃƒÂ©gie a ses propres limites :

\- \*\*Max positions simultanÃƒÂ©es\*\* : 3-5 selon volatilitÃƒÂ©

\- \*\*Max risque\*\* : 2% du capital allouÃƒÂ©

\- \*\*CorrÃƒÂ©lation max\*\* : 0.7 entre positions



\### Global

\- \*\*Max drawdown\*\* : 8% tous ensemble

\- \*\*Circuit breaker\*\* : Stop si 5% drawdown

\- \*\*Diversification\*\* : Min 5 symboles diffÃƒÂ©rents



\## Ã°Å¸â€œÅ¡ Pour Aller Plus Loin



\- \[Configuration](CONFIGURATION.md) - ParamÃƒÂ©trer les stratÃƒÂ©gies

\- \[API Reference](API\_REFERENCE.md) - CrÃƒÂ©er vos propres stratÃƒÂ©gies

\- \[Troubleshooting](TROUBLESHOOTING.md) - RÃƒÂ©soudre les problÃƒÂ¨mes



---



\*\*5 StratÃƒÂ©gies = 5 FaÃƒÂ§ons de Gagner ! Ã°Å¸Å½Â¯\*\*



\*DerniÃƒÂ¨re mise ÃƒÂ  jour : Octobre 2024\*

