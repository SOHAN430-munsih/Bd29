# Baseline vs ML Evaluator

Quick script to compare simple baselines (Median, Naive, SMA-k) vs a GradientBoostingRegressor on your multiplier time-series.

## Usage
```bash
pip install pandas numpy scikit-learn matplotlib
python evaluate_predictor.py --csv your_data.csv --sma_windows 5 10 20 --test_ratio 0.2
```

### CSV format
At minimum, a column named `multiplier`:
```
multiplier
1.5
2.0
1.2
5.0
1.8
...
```
