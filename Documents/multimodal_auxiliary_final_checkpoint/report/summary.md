# Auxiliary Task Case Study (No-Retrain)

- threshold: **0.95**
- max_batches: **0** (0 = full set)

## Key results
- Mean ΔIoU (new-old): **+0.0514**
- Median ΔIoU (new-old): **+0.0000**
- Fraction of samples where NEW > OLD (ΔIoU>0): **30.7%**

## Notes
- Landcover(aux) panels show **NEW model’s auxiliary landcover prediction**.
- Use `delta_iou_by_landcover.png` and `winrate_by_landcover.png` to discuss where auxiliary context helps.
- Use the top_improvements/top_regressions panels as qualitative evidence.
