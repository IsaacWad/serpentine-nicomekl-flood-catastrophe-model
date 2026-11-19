"""Summarize an existing event CSV without generating new flood events."""
import argparse
import json
from pathlib import Path


def summarize(input_path, output_dir):
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    events = pd.read_csv(input_path)
    # Failed runs contain zero placeholders; they must not enter loss statistics.
    good = events.loc[events['simulation_status'].eq('succeeded')].copy()
    failed = events.loc[~events['simulation_status'].eq('succeeded')]
    failed.to_csv(output_dir / 'failed_events.csv', index=False)
    summary = {'total_events':len(events), 'succeeded':len(good), 'failed_or_other':len(failed),
               'failure_or_other_fraction':len(failed)/len(events) if len(events) else None,
               'distribution':'gross building loss conditional on a simulated event succeeding'}
    if len(good):
        loss = pd.to_numeric(good['total_gross_loss'], errors='raise')
        if not np.isfinite(loss).all() or (loss < 0).any():
            raise ValueError('Successful event losses must be finite and non-negative')
        summary.update(mean_gross_loss=float(loss.mean()), median_gross_loss=float(loss.median()),
                       p95_gross_loss=float(loss.quantile(.95)), p99_gross_loss=float(loss.quantile(.99)),
                       max_gross_loss=float(loss.max()))
        good.groupby('event_source')['total_gross_loss'].agg(['count','mean','median','sum','max']).to_csv(output_dir / 'loss_by_source.csv')
        plt.figure(figsize=(8,5))
        plt.hist(np.log1p(loss), bins=60)
        plt.xlabel('log(1 + gross event loss)')
        plt.ylabel('Successful simulated events')
        plt.title('Conditional flood-event loss distribution')
        plt.tight_layout()
        plt.savefig(output_dir / 'conditional_loss_distribution.png', dpi=160)
        plt.close()
        ordered = np.sort(loss)[::-1]
        pd.DataFrame({'loss':ordered,'conditional_exceedance_probability':np.arange(1,len(loss)+1)/(len(loss)+1)}).to_csv(output_dir / 'conditional_exceedance.csv', index=False)
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument('--input', type=Path, default=root / 'results/event_results_clean_10000_master.csv')
    parser.add_argument('--output-dir', type=Path, default=root / 'outputs/archived_results_review')
    args = parser.parse_args()
    print(json.dumps(summarize(args.input, args.output_dir), indent=2))
