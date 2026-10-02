Strategic Properties for Text Filtering
• Dynamic Sensitivity Scaling: Instead of setting a rigid, arbitrary distance threshold (e.g., always flag above 3.0), the BH line changes dynamically with text size. If a long text block contains an unusually high amount of noise, the baseline line automatically tightens.
• Proportion Control: Setting \(q = 0.05\) bounds your system mathematically so that no more than 5% of your total anomaly alerts are expected to be false alarms on regular text blocks.
