# Model Comparison: Baseline Classification Pipeline (Telco Churn)

## Setup

- Dataset: Telco Customer Churn, 7,043 rows, 21 columns. Target: `Churn` (Yes = 1), churn rate 26.54%.
- `TotalCharges` was a string column with 11 blank values (new customers with tenure 0). Converted with `pd.to_numeric(errors="coerce")`; the 11 missing values are median-imputed inside the pipeline.
- `customerID` was dropped and is not used as a feature.
- Split: 80/20 with `stratify=y`, `random_state=42`. Train 5,634 rows, test 1,409 rows. Churn rate is 26.54% in both, so the split preserves the imbalance.
- Preprocessing (inside one `Pipeline`, fit on training data only): median imputation and `StandardScaler` for `tenure`, `MonthlyCharges`, `TotalCharges`; one-hot encoding for 15 categorical columns; `SeniorCitizen` passed through as 0/1.
- Models: majority-class baseline (Task 2), Logistic Regression with default weights, Logistic Regression with `class_weight='balanced'`. All evaluated on the same test set at the default 0.5 threshold.

## Metrics table (test set, Churn = positive class)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Task 2 Baseline (majority class) | 0.7346 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |
| LogReg (default weights) | 0.8055 | 0.6572 | 0.5588 | 0.6040 | 0.8419 |
| LogReg (class_weight='balanced') | 0.7381 | 0.5043 | 0.7834 | 0.6136 | 0.8414 |

## Confusion matrices (rows = actual, columns = predicted)

![Confusion matrices](confusion_matrices.png)

| | Pred No | Pred Yes |
|---|---|---|
| **Baseline: Actual No** | 1035 (TN) | 0 (FP) |
| **Baseline: Actual Yes** | 374 (FN) | 0 (TP) |

| | Pred No | Pred Yes |
|---|---|---|
| **Default: Actual No** | 926 (TN) | 109 (FP) |
| **Default: Actual Yes** | 165 (FN) | 209 (TP) |

| | Pred No | Pred Yes |
|---|---|---|
| **Balanced: Actual No** | 747 (TN) | 288 (FP) |
| **Balanced: Actual Yes** | 81 (FN) | 293 (TP) |

## Precision / recall trade-off

- The majority-class baseline gets 73.46% accuracy by predicting "No" for everyone. It never finds a single churner (recall 0), so its accuracy is meaningless for this problem. Both Logistic Regression models beat it clearly on ROC-AUC (0.84 vs 0.50).
- Default weights are optimised for overall accuracy, so the model leans toward predicting "No". It has the highest accuracy (80.55%) and precision (65.7%), but it misses 165 of 374 churners (recall 55.9%).
- `class_weight='balanced'` penalises mistakes on the minority class more heavily. Recall rises to 78.3% and missed churners drop to 81, but precision falls to 50.4% and false alarms rise from 109 to 288. Accuracy drops about 7 points, which is expected and not a sign of a worse model.
- ROC-AUC is almost identical (0.8419 vs 0.8414). Class weighting does not make the model rank customers better; it only moves the decision boundary. That is why accuracy and ROC-AUC alone would hide the real difference between the two versions. F1 is close (0.604 vs 0.614), so the choice has to come from business cost, not from F1.

## Threshold awareness

| Model | Threshold | Precision | Recall | F1 | Missed churners (FN) | False alarms (FP) |
|---|---|---|---|---|---|---|
| Default | 0.30 | 0.519 | 0.754 | 0.615 | 92 | 261 |
| Default | 0.40 | 0.568 | 0.668 | 0.614 | 124 | 190 |
| Default | 0.50 | 0.657 | 0.559 | 0.604 | 165 | 109 |
| Balanced | 0.30 | 0.430 | 0.928 | 0.587 | 27 | 461 |
| Balanced | 0.40 | 0.466 | 0.866 | 0.606 | 50 | 371 |
| Balanced | 0.50 | 0.504 | 0.783 | 0.614 | 81 | 288 |

Moving the threshold on the default model gives a similar recall/precision trade-off to class weighting (default at 0.30 has recall 0.754, close to balanced at 0.50 with 0.783). Class weighting and threshold tuning are two routes to the same goal of favouring recall.

## Business interpretation

A missed churner means a customer leaves with no retention offer, so the company loses the customer's future revenue. A false alarm means a retention offer (discount, call, upgrade) goes to someone who would have stayed, which costs far less. Since missing a churner is the more expensive error, recall matters more than precision here.

**Recommendation: deploy the `class_weight='balanced'` version.** Compared with the default model it catches 84 more churners out of 374 (293 vs 209) and cuts missed churners from 165 to 81. The price is 179 extra false alarms (288 vs 109). The balanced model is the better choice as long as one missed churner costs the business more than about 2.1 false-alarm offers (179 / 84 ≈ 2.1). A lost subscriber normally costs many months of revenue, far more than one retention offer, so this condition is very likely met.

Caveats: the actual cost ratio should be confirmed with the business. If retention offers are expensive or the retention team has limited capacity, the default model (or a threshold around 0.40) is a more practical choice because it produces fewer false alarms. If the team can follow up on many customers, a lower threshold on the balanced model (0.40) catches even more churners (recall 86.6%) at the cost of more false alarms.

Main churn drivers seen in the balanced model's coefficients: short tenure, month-to-month contracts and fiber-optic internet increase churn risk; long tenure and two-year contracts reduce it. This suggests retention efforts should focus on new, month-to-month customers.

## Reproduce

```
python classification_baseline.py
```

Outputs: `metrics_table.csv`, `threshold_sweep.csv`, `results.json`, `confusion_matrices.png`.
