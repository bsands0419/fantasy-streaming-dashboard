# Rookie Models Stage 5D: Feature Attribution and Disagreement Diagnosis

Generated 2026-08-19T14:06:38.855063+00:00

Stage 5D does not change the frozen model. It measures local sensitivity by replacing one current-player input, or one football feature family, with the historical same-position median and re-running the exact frozen Stage 3 regression and hit classifier. Positive deltas mean the observed value raises the player's frozen-model output relative to that neutral median. These are perturbation sensitivities, not causal effects and not additive SHAP values.

## Exact reproduction audit

|Pos|Current N|Union features|Regression max error|Hit-prob max error|Pass?|
|---|---:|---:|---:|---:|---|
|QB|10|125|1.78e-15|4.25e-09|Yes|
|RB|13|113|3.55e-15|1.19e-08|Yes|
|WR|36|371|1.78e-15|1.05e-08|Yes|
|TE|22|833|8.88e-16|1.29e-08|Yes|

## High-priority 2026 diagnoses

### Ty Simpson (QB)
- Grade: 89.2th percentile; projected best-2-of-3 PPG: 13.03; classifier hit: 43.2%.
- Stage 5C flag: signals broadly consistent. Confidence: Medium.
- Regression-positive families: Draft capital (+10.82 PPG); Rushing production (+2.61 PPG); Age / experience (+0.00 PPG).
- Regression-negative families: Passing production (-0.88 PPG).
- Classifier-positive families: Draft capital (+35.76 pp); Rushing production (+9.65 pp); Passing production (+0.37 pp).
- Classifier-negative families: none.
- Strongest positive score features: log draft pick (+3.06 PPG; Draft capital); draft pick (+2.70 PPG; Draft capital); inverse-root draft pick (+2.23 PPG; Draft capital); first-round indicator (+1.39 PPG; Draft capital); rushing share career trend (+0.49 PPG; Rushing production).
- Strongest negative score features: completion percentage experience-adjusted peak (-0.20 PPG; Passing production); EPAplay final season (-0.14 PPG; Passing production); QB rushing fantasy points per game career peak (-0.12 PPG; Rushing production); EPAplay career mean (-0.11 PPG; Passing production); completion percentage early-career peak (-0.09 PPG; Passing production).

### Carson Beck (QB)
- Grade: 73.3th percentile; projected best-2-of-3 PPG: 6.63; classifier hit: 5.9%.
- Stage 5C flag: signals broadly consistent. Confidence: Medium.
- Regression-positive families: Passing production (+3.44 PPG); Draft capital (+1.93 PPG); Rushing production (+0.29 PPG).
- Regression-negative families: none.
- Classifier-positive families: Rushing production (+0.93 pp); Draft capital (+0.09 pp).
- Classifier-negative families: Passing production (-0.15 pp).
- Strongest positive score features: completion percentage final season (+2.15 PPG; Passing production); day-two indicator (+0.56 PPG; Draft capital); log draft pick (+0.46 PPG; Draft capital); inverse-root draft pick (+0.46 PPG; Draft capital); success experience-adjusted peak (+0.29 PPG; Passing production).
- Strongest negative score features: EPAplay career peak (-0.09 PPG; Passing production); EPAplay final season (-0.06 PPG; Passing production); rushing share final season (-0.04 PPG; Rushing production); yards per game final season (-0.04 PPG; Rushing production); interception rate early-career peak (-0.04 PPG; Passing production).

### Drew Allar (QB)
- Grade: 63.1th percentile; projected best-2-of-3 PPG: 3.80; classifier hit: 10.0%.
- Stage 5C flag: signals broadly consistent. Confidence: High.
- Regression-positive families: Draft capital (+2.46 PPG); Rushing production (+0.66 PPG); Passing production (+0.45 PPG).
- Regression-negative families: Age / experience (-0.04 PPG).
- Classifier-positive families: Rushing production (+5.57 pp); Draft capital (+0.15 pp); Age / experience (+0.15 pp).
- Classifier-negative families: Passing production (-1.48 pp).
- Strongest positive score features: inverse-root draft pick (+0.65 PPG; Draft capital); log draft pick (+0.62 PPG; Draft capital); day-two indicator (+0.47 PPG; Draft capital); draft pick (+0.46 PPG; Draft capital); EPAplay experience-adjusted peak (+0.28 PPG; Passing production).
- Strongest negative score features: success final season (-0.05 PPG; Passing production); passing TD rate career trend (-0.05 PPG; Passing production); rushing share final season (-0.05 PPG; Rushing production); yards per dropback final season (-0.04 PPG; Passing production); draft age (-0.04 PPG; Age / experience).

### Taylen Green (QB)
- Grade: 39.8th percentile; projected best-2-of-3 PPG: 1.85; classifier hit: 11.0%.
- Stage 5C flag: out-of-distribution profile; low model familiarity. Confidence: Low.
- Regression-positive families: Rushing production (+1.20 PPG); Passing production (+0.24 PPG).
- Regression-negative families: Draft capital (-0.18 PPG); Age / experience (-0.13 PPG).
- Classifier-positive families: Rushing production (+6.63 pp).
- Classifier-negative families: Passing production (-1.46 pp).
- Strongest positive score features: yards per game career trend (+0.26 PPG; Rushing production); rushing share career trend (+0.17 PPG; Rushing production); QB rushing fantasy points per game career trend (+0.17 PPG; Rushing production); rushing share career trend (+0.16 PPG; Rushing production); yards per game career peak (+0.11 PPG; Rushing production).
- Strongest negative score features: inverse-root draft pick (-0.08 PPG; Draft capital); youth for position (-0.07 PPG; Age / experience); draft age (-0.07 PPG; Age / experience); draft pick (-0.06 PPG; Draft capital); EPAplay career mean (-0.05 PPG; Passing production).

### Cole Payton (QB)
- Grade: 25.6th percentile; projected best-2-of-3 PPG: 1.16; classifier hit: 5.0%.
- Stage 5C flag: out-of-distribution profile; low model familiarity. Confidence: Low.
- Regression-positive families: Passing production (+0.47 PPG); Rushing production (+0.23 PPG).
- Regression-negative families: Age / experience (-0.59 PPG); Draft capital (-0.15 PPG).
- Classifier-positive families: none.
- Classifier-negative families: none.
- Strongest positive score features: passing TD rate final season (+0.04 PPG; Passing production); pass college seasons (+0.04 PPG; Age / experience); interception rate experience-adjusted peak (+0.03 PPG; Passing production); passing TD rate career mean (+0.03 PPG; Passing production); success career mean (+0.02 PPG; Passing production).
- Strongest negative score features: draft age (-0.35 PPG; Age / experience); youth for position (-0.34 PPG; Age / experience); draft pick (-0.06 PPG; Draft capital); inverse-root draft pick (-0.04 PPG; Draft capital); log draft pick (-0.03 PPG; Draft capital).

### Jeremiyah Love (RB)
- Grade: 100.0th percentile; projected best-2-of-3 PPG: 16.06; classifier hit: 59.0%.
- Stage 5C flag: signals broadly consistent. Confidence: High.
- Regression-positive families: Draft capital (+11.02 PPG); Athletic profile (+0.37 PPG); Receiving production (+0.35 PPG).
- Regression-negative families: Rushing production (-0.08 PPG).
- Classifier-positive families: Draft capital (+45.55 pp); Athletic profile (+20.19 pp); Receiving production (+17.85 pp).
- Classifier-negative families: Rushing production (-3.53 pp).
- Strongest positive score features: log draft pick (+5.95 PPG; Draft capital); first-round indicator (+2.11 PPG; Draft capital); draft pick (+1.46 PPG; Draft capital); inverse-root draft pick (+1.02 PPG; Draft capital); draft round (+0.47 PPG; Draft capital).
- Strongest negative score features: success experience-adjusted peak (-0.29 PPG; Receiving production); vertical (-0.27 PPG; Athletic profile); scrimmage yards per game career peak (-0.19 PPG; Rushing production); rushing-yard share career trend (-0.13 PPG; Rushing production); burst proxy (-0.07 PPG; Athletic profile).

### Jadarian Price (RB)
- Grade: 96.8th percentile; projected best-2-of-3 PPG: 12.18; classifier hit: 25.5%.
- Stage 5C flag: classifier materially below bucket hit rate; five comps materially below bucket hit rate; elite grade but modest classifier probability. Confidence: Medium.
- Regression-positive families: Draft capital (+6.17 PPG); Receiving production (+1.52 PPG); Athletic profile (+0.49 PPG).
- Regression-negative families: Rushing production (-0.37 PPG); Age / experience (-0.00 PPG).
- Classifier-positive families: Draft capital (+21.42 pp); Athletic profile (+6.41 pp); Receiving production (+5.23 pp).
- Classifier-negative families: Rushing production (-5.19 pp).
- Strongest positive score features: log draft pick (+2.26 PPG; Draft capital); first-round indicator (+2.11 PPG; Draft capital); draft pick (+1.14 PPG; Draft capital); EPAplay career peak (+0.63 PPG; Receiving production); draft round (+0.47 PPG; Draft capital).
- Strongest negative score features: scrimmage yards per game early-career peak (-0.50 PPG; Rushing production); target share final season (-0.41 PPG; Receiving production); success experience-adjusted peak (-0.24 PPG; Receiving production); scrimmage yards per game career mean (-0.18 PPG; Rushing production); target share early-career peak (-0.15 PPG; Receiving production).

### Eli Heidenreich (RB)
- Grade: 59.8th percentile; projected best-2-of-3 PPG: 4.77; classifier hit: 13.9%.
- Stage 5C flag: out-of-distribution profile; low model familiarity; five comps materially above bucket hit rate. Confidence: Low.
- Regression-positive families: Receiving production (+1.57 PPG); Rushing production (+0.79 PPG); Athletic profile (+0.39 PPG).
- Regression-negative families: Draft capital (-2.36 PPG); Age / experience (-0.01 PPG).
- Classifier-positive families: Receiving production (+9.39 pp); Athletic profile (+4.28 pp); Rushing production (+2.30 pp).
- Classifier-negative families: none.
- Strongest positive score features: target share final season (+2.60 PPG; Receiving production); target share early-career peak (+0.79 PPG; Receiving production); EPAplay career peak (+0.57 PPG; Receiving production); success early-career peak (+0.51 PPG; Rushing production); success career mean (+0.33 PPG; Receiving production).
- Strongest negative score features: target share experience-adjusted peak (-1.17 PPG; Receiving production); draft pick (-1.03 PPG; Draft capital); target share career mean (-1.01 PPG; Receiving production); log draft pick (-0.82 PPG; Draft capital); target share career peak (-0.69 PPG; Receiving production).

### Adam Randall (RB)
- Grade: 34.9th percentile; projected best-2-of-3 PPG: 2.93; classifier hit: 4.1%.
- Stage 5C flag: out-of-distribution profile; low model familiarity. Confidence: Low.
- Regression-positive families: Athletic profile (+0.22 PPG).
- Regression-negative families: Draft capital (-0.98 PPG); Rushing production (-0.33 PPG); Age / experience (-0.31 PPG).
- Classifier-positive families: Rushing production (+1.25 pp); Athletic profile (+0.74 pp); Age / experience (+0.70 pp).
- Classifier-negative families: Receiving production (-1.49 pp).
- Strongest positive score features: scrimmage yards per game early-career peak (+1.08 PPG; Rushing production); rushing share early-career peak (+0.40 PPG; Rushing production); target share final season (+0.28 PPG; Receiving production); broad jump (+0.27 PPG; Athletic profile); shuttle (+0.27 PPG; Athletic profile).
- Strongest negative score features: success early-career peak (-0.73 PPG; Rushing production); scrimmage yards per game career peak (-0.54 PPG; Rushing production); rushing-yard share career trend (-0.45 PPG; Rushing production); draft pick (-0.42 PPG; Draft capital); log draft pick (-0.38 PPG; Draft capital).

### Kenyon Sadiq (TE)
- Grade: 96.1th percentile; projected best-2-of-3 PPG: 8.23; classifier hit: 36.6%.
- Stage 5C flag: classifier materially below bucket hit rate. Confidence: Medium.
- Regression-positive families: Draft capital (+4.02 PPG); College team context (+0.60 PPG); Scouting surprise (+0.02 PPG).
- Regression-negative families: Athletic profile (-0.02 PPG).
- Classifier-positive families: Draft capital (+29.07 pp); Athletic profile (+5.26 pp); Receiving production (+2.76 pp).
- Classifier-negative families: none.
- Strongest positive score features: draft pick (+1.68 PPG; Draft capital); log draft pick (+1.20 PPG; Draft capital); draft round (+0.62 PPG; Draft capital); inverse-root draft pick (+0.54 PPG; Draft capital); target share final season (+0.17 PPG; Receiving production).
- Strongest negative score features: targets early-career peak (-0.14 PPG; Receiving production); receiving production per team pass attempt early-career peak (-0.12 PPG; Receiving production); targets per game early-career peak (-0.03 PPG; Receiving production); shuttle (-0.02 PPG; Athletic profile); EPAgame experience-adjusted peak (-0.02 PPG; Receiving production).

### Eli Stowers (TE)
- Grade: 86.6th percentile; projected best-2-of-3 PPG: 6.31; classifier hit: 37.1%.
- Stage 5C flag: out-of-distribution profile; low model familiarity. Confidence: Low.
- Regression-positive families: Draft capital (+1.65 PPG); Receiving production (+0.50 PPG); Athletic profile (+0.48 PPG).
- Regression-negative families: Age / experience (-0.01 PPG); College team context (-0.01 PPG).
- Classifier-positive families: Draft capital (+27.50 pp); Athletic profile (+5.98 pp); Receiving production (+4.77 pp).
- Classifier-negative families: none.
- Strongest positive score features: log draft pick (+0.65 PPG; Draft capital); draft round (+0.56 PPG; Draft capital); draft pick (+0.55 PPG; Draft capital); shuttle (+0.36 PPG; Athletic profile); inverse-root draft pick (+0.23 PPG; Draft capital).
- Strongest negative score features: catchpct experience-adjusted peak (-0.10 PPG; Receiving production); day-two indicator (-0.07 PPG; Draft capital); team passrate off career trend (-0.06 PPG; College team context); team rushrate off career trend (-0.04 PPG; College team context); receiving-yard share career peak (-0.03 PPG; Receiving production).

### Tanner Koziol (TE)
- Grade: 68.5th percentile; projected best-2-of-3 PPG: 4.12; classifier hit: 9.5%.
- Stage 5C flag: signals broadly consistent. Confidence: Medium.
- Regression-positive families: Receiving production (+1.09 PPG); Athletic profile (+0.20 PPG); Other production (+0.07 PPG).
- Regression-negative families: Draft capital (-0.43 PPG); Scouting surprise (-0.00 PPG).
- Classifier-positive families: Receiving production (+4.66 pp); Athletic profile (+2.22 pp).
- Classifier-negative families: Draft capital (-2.90 pp).
- Strongest positive score features: rec td per game early-career peak (+0.11 PPG; Receiving production); vertical (+0.10 PPG; Athletic profile); receptions per team pass attempt experience-adjusted peak (+0.10 PPG; Receiving production); receiving-TD share experience-adjusted peak (+0.09 PPG; Receiving production); receiving-TD share career mean (+0.08 PPG; Receiving production).
- Strongest negative score features: draft round (-0.32 PPG; Draft capital); draft pick (-0.05 PPG; Draft capital); team adj off epa early-career peak (-0.02 PPG; College team context); log draft pick (-0.02 PPG; Draft capital); height inches (-0.02 PPG; Athletic profile).

### Sam Roush (TE)
- Grade: 68.1th percentile; projected best-2-of-3 PPG: 4.04; classifier hit: 41.6%.
- Stage 5C flag: classifier materially above bucket hit rate. Confidence: High.
- Regression-positive families: Draft capital (+1.12 PPG); Scouting surprise (+0.04 PPG).
- Regression-negative families: Receiving production (-1.06 PPG); College team context (-0.16 PPG); Other production (-0.08 PPG).
- Classifier-positive families: Draft capital (+19.19 pp); Athletic profile (+18.18 pp); Receiving production (+16.28 pp).
- Classifier-negative families: none.
- Strongest positive score features: draft pick (+0.47 PPG; Draft capital); log draft pick (+0.35 PPG; Draft capital); inverse-root draft pick (+0.17 PPG; Draft capital); vertical (+0.05 PPG; Athletic profile); scout expected log pick (+0.03 PPG; Scouting surprise).
- Strongest negative score features: catchpct experience-adjusted peak (-0.21 PPG; Receiving production); EPAgame final season (-0.13 PPG; Receiving production); receiving-TD share career mean (-0.11 PPG; Receiving production); weight lbs (-0.10 PPG; Athletic profile); day-two indicator (-0.06 PPG; Draft capital).

### Nate Boerkircher (TE)
- Grade: 62.1th percentile; projected best-2-of-3 PPG: 3.62; classifier hit: 22.0%.
- Stage 5C flag: signals broadly consistent. Confidence: High.
- Regression-positive families: Draft capital (+1.23 PPG); Athletic profile (+0.18 PPG); Other production (+0.03 PPG).
- Regression-negative families: Receiving production (-1.87 PPG); Age / experience (-0.01 PPG).
- Classifier-positive families: Draft capital (+17.82 pp); Athletic profile (+3.46 pp).
- Classifier-negative families: Receiving production (-9.33 pp).
- Strongest positive score features: draft pick (+0.47 PPG; Draft capital); log draft pick (+0.42 PPG; Draft capital); draft round (+0.16 PPG; Draft capital); inverse-root draft pick (+0.15 PPG; Draft capital); cone (+0.05 PPG; Athletic profile).
- Strongest negative score features: day-two indicator (-0.15 PPG; Draft capital); success career peak (-0.15 PPG; Receiving production); success experience-adjusted peak (-0.14 PPG; Receiving production); success early-career peak (-0.11 PPG; Receiving production); receiving production per team pass attempt early-career peak (-0.07 PPG; Receiving production).

### Jaren Kanak (TE)
- Grade: 5.6th percentile; projected best-2-of-3 PPG: 0.48; classifier hit: 11.5%.
- Stage 5C flag: out-of-distribution profile; low model familiarity. Confidence: Low.
- Regression-positive families: College team context (+0.20 PPG); Receiving production (+0.08 PPG); Athletic profile (+0.04 PPG).
- Regression-negative families: Draft capital (-3.72 PPG).
- Classifier-positive families: Receiving production (+6.23 pp); Athletic profile (+3.36 pp).
- Classifier-negative families: Draft capital (-3.53 pp).
- Strongest positive score features: team def strength faced career peak (+0.05 PPG; College team context); team def strength faced experience-adjusted peak (+0.03 PPG; College team context); forty (+0.03 PPG; Athletic profile); team EPAplay off pass career mean (+0.03 PPG; College team context); team def strength faced final season (+0.01 PPG; College team context).
- Strongest negative score features: draft round (-1.77 PPG; Draft capital); draft pick (-1.22 PPG; Draft capital); log draft pick (-0.91 PPG; Draft capital); inverse-root draft pick (-0.81 PPG; Draft capital); team yardsplay off experience-adjusted peak (-0.01 PPG; College team context).

### KC Concepcion (WR)
- Grade: 91.8th percentile; projected best-2-of-3 PPG: 9.92; classifier hit: 67.3%.
- Stage 5C flag: classifier materially above bucket hit rate; five comps materially below bucket hit rate. Confidence: Medium.
- Regression-positive families: Draft capital (+4.62 PPG); Receiving production (+1.38 PPG); Scouting surprise (+0.17 PPG).
- Regression-negative families: Age / experience (-0.12 PPG).
- Classifier-positive families: Draft capital (+43.91 pp); Receiving production (+29.45 pp); Scouting surprise (+5.49 pp).
- Classifier-negative families: Other production (-0.04 pp).
- Strongest positive score features: draft pick (+1.38 PPG; Draft capital); draft round (+1.22 PPG; Draft capital); log draft pick (+0.83 PPG; Draft capital); first-round indicator (+0.83 PPG; Draft capital); rec td per game experience-adjusted peak (+0.59 PPG; Receiving production).
- Strongest negative score features: rec td per game early-career peak (-0.15 PPG; Receiving production); success career mean (-0.12 PPG; Receiving production); EPAplay experience-adjusted peak (-0.09 PPG; Receiving production); youth for position (-0.08 PPG; Age / experience); yards per target experience-adjusted peak (-0.06 PPG; Receiving production).

### De'Zhaun Stribling (WR)
- Grade: 78.9th percentile; projected best-2-of-3 PPG: 7.56; classifier hit: 14.9%.
- Stage 5C flag: signals broadly consistent. Confidence: High.
- Regression-positive families: Draft capital (+4.94 PPG); Receiving production (+1.03 PPG); Other production (+0.15 PPG).
- Regression-negative families: Age / experience (-0.08 PPG).
- Classifier-positive families: Draft capital (+8.47 pp); Scouting surprise (+4.08 pp); Receiving production (+1.23 pp).
- Classifier-negative families: Age / experience (-4.96 pp).
- Strongest positive score features: draft round (+1.53 PPG; Draft capital); draft pick (+0.78 PPG; Draft capital); log draft pick (+0.69 PPG; Draft capital); day-two indicator (+0.60 PPG; Draft capital); inverse-root draft pick (+0.40 PPG; Draft capital).
- Strongest negative score features: yards per play experience-adjusted peak (-0.15 PPG; Receiving production); EPAplay early-career peak (-0.06 PPG; Receiving production); targets career trend (-0.05 PPG; Receiving production); yards per play early-career peak (-0.05 PPG; Receiving production); scout expected log pick (-0.05 PPG; Scouting surprise).

### Jordyn Tyson (WR)
- Grade: 77.7th percentile; projected best-2-of-3 PPG: 7.35; classifier hit: 29.2%.
- Stage 5C flag: five comps materially above bucket hit rate. Confidence: Medium.
- Regression-positive families: Draft capital (+4.14 PPG); Scouting surprise (+0.14 PPG); Other production (+0.08 PPG).
- Regression-negative families: Receiving production (-0.73 PPG); Age / experience (-0.12 PPG).
- Classifier-positive families: Draft capital (+24.42 pp); Scouting surprise (+7.87 pp); Other production (+0.78 pp).
- Classifier-negative families: Age / experience (-7.33 pp); Receiving production (-0.38 pp).
- Strongest positive score features: draft pick (+1.01 PPG; Draft capital); draft round (+0.87 PPG; Draft capital); log draft pick (+0.75 PPG; Draft capital); EPAgame early-career peak (+0.30 PPG; Receiving production); inverse-root draft pick (+0.26 PPG; Draft capital).
- Strongest negative score features: yards per play early-career peak (-0.43 PPG; Receiving production); yards per target early-career peak (-0.41 PPG; Receiving production); yards per play experience-adjusted peak (-0.19 PPG; Receiving production); yards per target career peak (-0.17 PPG; Receiving production); EPAplay experience-adjusted peak (-0.17 PPG; Receiving production).

### Omar Cooper Jr. (WR)
- Grade: 69.1th percentile; projected best-2-of-3 PPG: 5.96; classifier hit: 7.0%.
- Stage 5C flag: signals broadly consistent. Confidence: Medium.
- Regression-positive families: Draft capital (+2.11 PPG); Scouting surprise (+0.56 PPG); Other production (+0.13 PPG).
- Regression-negative families: Receiving production (-2.78 PPG).
- Classifier-positive families: Draft capital (+4.84 pp); Scouting surprise (+2.19 pp); Other production (+0.24 pp).
- Classifier-negative families: Receiving production (-12.61 pp).
- Strongest positive score features: scout expected log pick (+0.55 PPG; Scouting surprise); first-round indicator (+0.46 PPG; Draft capital); draft round (+0.44 PPG; Draft capital); draft pick (+0.41 PPG; Draft capital); log draft pick (+0.29 PPG; Draft capital).
- Strongest negative score features: yards per target early-career peak (-0.45 PPG; Receiving production); yards per play early-career peak (-0.45 PPG; Receiving production); yards per play experience-adjusted peak (-0.30 PPG; Receiving production); receptions per game career mean (-0.29 PPG; Receiving production); yards per target experience-adjusted peak (-0.19 PPG; Receiving production).

### Germie Bernard (WR)
- Grade: 66.9th percentile; projected best-2-of-3 PPG: 5.66; classifier hit: 7.2%.
- Stage 5C flag: signals broadly consistent. Confidence: Medium.
- Regression-positive families: Draft capital (+2.84 PPG); Other production (+0.18 PPG); Scouting surprise (+0.08 PPG).
- Regression-negative families: Receiving production (-0.33 PPG); Age / experience (-0.08 PPG).
- Classifier-positive families: Draft capital (+4.68 pp); Scouting surprise (+2.34 pp); Other production (+0.25 pp).
- Classifier-negative families: Receiving production (-6.64 pp); Age / experience (-2.15 pp).
- Strongest positive score features: draft round (+1.23 PPG; Draft capital); day-two indicator (+0.50 PPG; Draft capital); log draft pick (+0.25 PPG; Draft capital); draft pick (+0.21 PPG; Draft capital); season combine (+0.17 PPG; Other production).
- Strongest negative score features: td per target early-career peak (-0.25 PPG; Receiving production); td per target career peak (-0.24 PPG; Receiving production); EPAplay experience-adjusted peak (-0.19 PPG; Receiving production); td per target career mean (-0.15 PPG; Receiving production); td per target experience-adjusted peak (-0.13 PPG; Receiving production).

### Elijah Sarratt (WR)
- Grade: 65.0th percentile; projected best-2-of-3 PPG: 5.43; classifier hit: 14.7%.
- Stage 5C flag: out-of-distribution profile; low model familiarity. Confidence: Low.
- Regression-positive families: Receiving production (+1.88 PPG); Scouting surprise (+0.44 PPG); Age / experience (+0.06 PPG).
- Regression-negative families: Other production (-0.13 PPG).
- Classifier-positive families: Receiving production (+10.47 pp); Scouting surprise (+1.46 pp); Other production (+0.28 pp).
- Classifier-negative families: none.
- Strongest positive score features: yards per game career mean (+0.78 PPG; Receiving production); receptions per game career mean (+0.39 PPG; Receiving production); scout expected log pick (+0.32 PPG; Scouting surprise); receiving-yard share career mean (+0.23 PPG; Receiving production); reception share career mean (+0.19 PPG; Receiving production).
- Strongest negative score features: success experience-adjusted peak (-0.22 PPG; Receiving production); weight (-0.12 PPG; Other production); success career peak (-0.12 PPG; Receiving production); success early-career peak (-0.10 PPG; Receiving production); EPAgame early-career peak (-0.07 PPG; Receiving production).

### Malachi Fields (WR)
- Grade: 65.0th percentile; projected best-2-of-3 PPG: 5.40; classifier hit: 6.0%.
- Stage 5C flag: signals broadly consistent. Confidence: Medium.
- Regression-positive families: Draft capital (+3.16 PPG); Other production (+0.29 PPG); Age / experience (+0.07 PPG).
- Regression-negative families: Receiving production (-0.40 PPG).
- Classifier-positive families: Draft capital (+3.30 pp); Scouting surprise (+0.27 pp); Other production (+0.26 pp).
- Classifier-negative families: Age / experience (-2.01 pp); Receiving production (-1.89 pp).
- Strongest positive score features: day-two indicator (+0.84 PPG; Draft capital); draft round (+0.82 PPG; Draft capital); draft pick (+0.49 PPG; Draft capital); inverse-root draft pick (+0.35 PPG; Draft capital); season combine (+0.23 PPG; Other production).
- Strongest negative score features: EPAplay early-career peak (-0.15 PPG; Receiving production); passing td career mean (-0.11 PPG; Receiving production); receiving-TD share final season (-0.09 PPG; Receiving production); passing td final season (-0.08 PPG; Receiving production); td per target early-career peak (-0.07 PPG; Receiving production).

### Ted Hurst (WR)
- Grade: 61.5th percentile; projected best-2-of-3 PPG: 4.92; classifier hit: 22.6%.
- Stage 5C flag: out-of-distribution profile; low model familiarity. Confidence: Low.
- Regression-positive families: Draft capital (+0.37 PPG); Scouting surprise (+0.17 PPG).
- Regression-negative families: Receiving production (-0.46 PPG); Age / experience (-0.07 PPG); Other production (-0.00 PPG).
- Classifier-positive families: Draft capital (+13.92 pp); Receiving production (+7.76 pp); Scouting surprise (+4.20 pp).
- Classifier-negative families: Other production (-0.22 pp).
- Strongest positive score features: receiving-yard share career trend (+0.20 PPG; Receiving production); draft pick (+0.15 PPG; Draft capital); reception share career mean (+0.14 PPG; Receiving production); targets experience-adjusted peak (+0.13 PPG; Receiving production); passing td early-career peak (+0.12 PPG; Receiving production).
- Strongest negative score features: day-two indicator (-0.20 PPG; Draft capital); passing td final season (-0.16 PPG; Receiving production); catchpct career peak (-0.15 PPG; Receiving production); catchpct early-career peak (-0.11 PPG; Receiving production); catchpct career trend (-0.10 PPG; Receiving production).

### Chris Brazzell II (WR)
- Grade: 56.2th percentile; projected best-2-of-3 PPG: 4.12; classifier hit: 18.7%.
- Stage 5C flag: five comps materially above bucket hit rate. Confidence: High.
- Regression-positive families: Draft capital (+1.26 PPG); Age / experience (+0.05 PPG); Other production (+0.04 PPG).
- Regression-negative families: Receiving production (-0.66 PPG).
- Classifier-positive families: Draft capital (+10.54 pp); Receiving production (+7.65 pp); Scouting surprise (+0.64 pp).
- Classifier-negative families: none.
- Strongest positive score features: day-two indicator (+0.37 PPG; Draft capital); log draft pick (+0.28 PPG; Draft capital); receiving-yard share career trend (+0.23 PPG; Receiving production); draft pick (+0.22 PPG; Draft capital); draft round (+0.18 PPG; Draft capital).
- Strongest negative score features: rec td per game early-career peak (-0.17 PPG; Receiving production); receiving-TD share experience-adjusted peak (-0.09 PPG; Receiving production); target share experience-adjusted peak (-0.09 PPG; Receiving production); reception share experience-adjusted peak (-0.09 PPG; Receiving production); reception share career peak (-0.08 PPG; Receiving production).

### Bryce Lance (WR)
- Grade: 40.8th percentile; projected best-2-of-3 PPG: 3.03; classifier hit: 2.9%.
- Stage 5C flag: out-of-distribution profile; low model familiarity. Confidence: Low.
- Regression-positive families: Receiving production (+0.79 PPG).
- Regression-negative families: Draft capital (-0.65 PPG); Scouting surprise (-0.22 PPG); Age / experience (-0.21 PPG).
- Classifier-positive families: Age / experience (+0.06 pp).
- Classifier-negative families: Draft capital (-0.87 pp); Scouting surprise (-0.04 pp).
- Strongest positive score features: receiving-TD share career trend (+0.11 PPG; Receiving production); receiving-TD share early-career peak (+0.07 PPG; Receiving production); EPAplay career peak (+0.04 PPG; Receiving production); targets per game final season (+0.03 PPG; Receiving production); playsgame early-career peak (+0.02 PPG; Receiving production).
- Strongest negative score features: draft pick (-0.24 PPG; Draft capital); inverse-root draft pick (-0.23 PPG; Draft capital); scout expected log pick (-0.22 PPG; Scouting surprise); log draft pick (-0.20 PPG; Draft capital); youth for position (-0.13 PPG; Age / experience).

### Kaden Wetjen (WR)
- Grade: 38.1th percentile; projected best-2-of-3 PPG: 2.86; classifier hit: 2.0%.
- Stage 5C flag: out-of-distribution profile; low model familiarity. Confidence: Low.
- Regression-positive families: Scouting surprise (+0.15 PPG); Other production (+0.07 PPG).
- Regression-negative families: Receiving production (-0.46 PPG); Age / experience (-0.13 PPG); Draft capital (-0.04 PPG).
- Classifier-positive families: Scouting surprise (+0.10 pp); Other production (+0.07 pp); Age / experience (+0.04 pp).
- Classifier-negative families: Receiving production (-1.16 pp); Draft capital (-0.49 pp).
- Strongest positive score features: season combine (+0.15 PPG; Other production); scout boost (+0.10 PPG; Scouting surprise); receptions per game career mean (+0.08 PPG; Receiving production); passing td career peak (+0.08 PPG; Receiving production); td per target experience-adjusted peak (+0.06 PPG; Receiving production).
- Strongest negative score features: EPAplay career trend (-0.39 PPG; Receiving production); target share early-career peak (-0.12 PPG; Receiving production); EPAgame career trend (-0.12 PPG; Receiving production); success career trend (-0.09 PPG; Receiving production); EPAplay final season (-0.09 PPG; Receiving production).

### Colbie Young (WR)
- Grade: 25.7th percentile; projected best-2-of-3 PPG: 2.12; classifier hit: 4.3%.
- Stage 5C flag: five comps materially above bucket hit rate. Confidence: High.
- Regression-positive families: Other production (+0.07 PPG); Scouting surprise (+0.00 PPG).
- Regression-negative families: Receiving production (-0.45 PPG); Draft capital (-0.41 PPG); Age / experience (-0.04 PPG).
- Classifier-positive families: Scouting surprise (+1.26 pp); Receiving production (+0.96 pp); Other production (+0.15 pp).
- Classifier-negative families: Draft capital (-0.85 pp); Age / experience (-0.80 pp).
- Strongest positive score features: season combine (+0.12 PPG; Other production); receiving-yard share early-career peak (+0.06 PPG; Receiving production); EPAgame final season (+0.05 PPG; Receiving production); td per target career trend (+0.05 PPG; Receiving production); receiving-yard share experience-adjusted peak (+0.04 PPG; Receiving production).
- Strongest negative score features: draft pick (-0.16 PPG; Draft capital); log draft pick (-0.12 PPG; Draft capital); inverse-root draft pick (-0.11 PPG; Draft capital); EPAgame career trend (-0.08 PPG; Receiving production); reception share experience-adjusted peak (-0.08 PPG; Receiving production).

## Interpretation guardrails

- A positive local delta means that neutralizing that observed input to the historical position median lowers the output. It does not prove the feature causes NFL success.
- Tree-model interactions mean single-feature deltas do not sum to the full prediction. Family-level neutralization is included to expose interaction-sensitive football themes.
- Missing-value indicators can matter. When a current input is missing, replacing it with the historical median measures both value imputation and removal of the model's missingness signal.
- Regression grade and hit classifier are intentionally explained separately because Stage 5C showed meaningful disagreement for several prospects.

## Selected feature-family audit

This audit uses the corrected football taxonomy: draft age is age/experience; QB-derived rushing features are rushing; receiving-per-team-pass metrics remain receiving production; explicit pass/rush/rec team-context fields remain college context.

|Pos|Family|Union N|Primary reg|Secondary reg|Hit clf|
|---|---|---:|---:|---:|---:|
|QB|Age / experience|17|17|17|17|
|QB|Draft capital|6|6|6|6|
|QB|Passing production|42|42|42|42|
|QB|Receiving production|12|12|12|12|
|QB|Rushing production|48|48|48|48|
|RB|Age / experience|17|17|0|17|
|RB|Athletic profile|12|12|0|12|
|RB|Draft capital|6|6|6|6|
|RB|Passing production|12|12|0|12|
|RB|Receiving production|18|18|0|18|
|RB|Rushing production|48|48|0|48|
|WR|Age / experience|17|17|0|17|
|WR|Draft capital|6|6|0|6|
|WR|Other production|4|4|0|4|
|WR|Passing production|102|102|0|102|
|WR|Receiving production|114|114|0|114|
|WR|Rushing production|126|126|0|126|
|WR|Scouting surprise|2|2|0|2|
|TE|Age / experience|17|17|17|17|
|TE|Athletic profile|12|12|12|12|
|TE|College team context|426|0|426|0|
|TE|Draft capital|6|6|6|6|
|TE|Other production|4|0|4|0|
|TE|Passing production|102|18|102|18|
|TE|Receiving production|126|48|126|48|
|TE|Rushing production|138|36|138|36|
|TE|Scouting surprise|2|0|2|0|

## Scouting-surprise membership audit

The rows below are exact membership in the frozen jobs. Their presence is diagnostic, not a Stage 5D model change.

|Pos|Feature|Primary reg?|Secondary reg?|Hit clf?|Primary blend weight|
|---|---|---:|---:|---:|---:|
|WR|scout_expected_log_pick|1|0|1|1.00|
|WR|scout_boost|1|0|1|1.00|
|TE|scout_expected_log_pick|0|1|0|0.20|
|TE|scout_boost|0|1|0|0.20|

## Effective frozen-feature audit

Frozen job feature lists contain some position-irrelevant columns that were entirely missing for that position and were skipped by scikit-learn's median imputer. The table below separates stored names from historically observed inputs. `Stage5B usable` additionally requires at least 20 historical observations and non-zero variance, matching the similarity-space rule.

|Pos|Family|Stored|Observed|Stage5B usable|Primary observed|Secondary observed|Hit-clf observed|
|---|---|---:|---:|---:|---:|---:|---:|
|QB|Age / experience|17|11|11|11|11|11|
|QB|Draft capital|6|6|6|6|6|6|
|QB|Passing production|42|42|42|42|42|42|
|QB|Receiving production|12|0|0|0|0|0|
|QB|Rushing production|48|48|48|48|48|48|
|RB|Age / experience|17|11|11|11|0|11|
|RB|Athletic profile|12|12|12|12|0|12|
|RB|Draft capital|6|6|6|6|6|6|
|RB|Passing production|12|0|0|0|0|0|
|RB|Receiving production|18|18|18|18|0|18|
|RB|Rushing production|48|30|30|30|0|30|
|WR|Age / experience|17|5|5|5|0|5|
|WR|Draft capital|6|6|6|6|0|6|
|WR|Other production|4|4|4|4|0|4|
|WR|Passing production|102|0|0|0|0|0|
|WR|Receiving production|114|114|114|114|0|114|
|WR|Rushing production|126|0|0|0|0|0|
|WR|Scouting surprise|2|2|2|2|0|2|
|TE|Age / experience|17|5|5|5|5|5|
|TE|Athletic profile|12|12|12|12|12|12|
|TE|College team context|426|156|156|0|156|0|
|TE|Draft capital|6|6|6|6|6|6|
|TE|Other production|4|4|4|0|4|0|
|TE|Passing production|102|0|0|0|0|0|
|TE|Receiving production|126|126|126|48|126|48|
|TE|Rushing production|138|0|0|0|0|0|
|TE|Scouting surprise|2|2|2|0|2|0|

### Effective scouting-surprise coverage

|Pos|Feature|Primary?|Secondary?|Hit clf?|Historical N|2026 N|Primary blend wt|
|---|---|---:|---:|---:|---:|---:|---:|
|WR|scout_expected_log_pick|1|0|1|525|36|1.00|
|WR|scout_boost|1|0|1|525|36|1.00|
|TE|scout_expected_log_pick|0|1|0|216|22|0.20|
|TE|scout_boost|0|1|0|216|22|0.20|
