# meaning (Metrics Interpretation)

## Strategy Used

To isolate and evaluate how your anomaly detection threshold manages text outliers, we can calculate and plot a Receiver Operating Characteristic (ROC) curve and a Precision-Recall (PR) curve using pure Matplotlib and Scikit-Learn.
This implementation converts multi-class distances into binary anomalies (1 = Anomaly, 0 = Normal Proximity) and sweeps across continuous distance scores to chart true baseline behavior against precision `drop-offs`.

## ROC AUC ({roc_auc:.2f}):

 Quantifies the overall probability that a randomly chosen text anomaly will be ranked with a higher distance score than a normal word.
##  Average Precision ({avg_precision:.2f}):
 Summarizes the PR curve profile, offering a clearer diagnostic signal when dealing with severely imbalanced text data (where normal words outnumber anomalies).