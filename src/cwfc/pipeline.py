"""Reproducible, training-only model selection for the Atomity challenge."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

TARGETS = ['active_vm_count', 'aggregate_cpu_utilization']
# Fixed candidate set, declared before final labels are evaluated.
CANDIDATES = ['last', 'mean6', 'mean12', 'mean24', 'ses02', 'ses05', 'damped', 'ar_ridge']


def verify_checksums(data: Path) -> dict:
    result = {}
    for line in (data / 'SHA256SUMS').read_text().splitlines():
        if not line.strip():
            continue
        expected, name = line.split(maxsplit=1)
        name = name.strip().lstrip('*')
        file = data / name
        if file.resolve().parent != data.resolve():
            raise ValueError('Unsafe checksum path')
        actual = hashlib.sha256(file.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Checksum failed: {name}')
        if name in result:
            raise ValueError('Duplicate checksum entry')
        result[name] = actual
    if not {'vm_cpu_readings.csv.gz', 'vm_metadata.csv.gz', 'DATA_CARD.md'}.issubset(result):
        raise ValueError('Incomplete checksum manifest')
    return result


def load_data(data: Path):
    raw = pd.read_csv(data / 'vm_cpu_readings.csv.gz')
    meta = pd.read_csv(data / 'vm_metadata.csv.gz')
    required = {'timestamp', 'vm_id', 'avg_cpu', 'min_cpu', 'max_cpu'}
    if not required.issubset(raw):
        raise ValueError('Missing CPU columns')
    if raw[['timestamp', 'vm_id']].isna().any().any():
        raise ValueError('Missing timestamp or VM identifier')
    if raw.duplicated(['timestamp', 'vm_id']).any():
        raise ValueError('Duplicate VM/timestamp pairs; refuse ambiguous weighting')
    if meta.vm_id.isna().any() or meta.vm_id.duplicated().any():
        raise ValueError('Metadata identifiers must be unique and nonmissing')
    if set(raw.vm_id) != set(meta.vm_id):
        raise ValueError('CPU and metadata populations differ')
    for col in ['timestamp', 'avg_cpu', 'min_cpu', 'max_cpu']:
        raw[col] = pd.to_numeric(raw[col], errors='raise')
    if not np.isfinite(raw.timestamp).all():
        raise ValueError('Nonfinite timestamps')
    valid = np.isfinite(raw.avg_cpu) & raw.avg_cpu.between(0, 100)
    # A valid observation means finite average CPU in [0,100]. Invalid rows are
    # excluded from BOTH targets, never silently filled with zero.
    aggregate = raw.loc[valid].groupby('timestamp').agg(
        active_vm_count=('vm_id', 'nunique'),
        aggregate_cpu_utilization=('avg_cpu', 'mean')).sort_index()
    times = np.sort(raw.timestamp.unique())
    aggregate = aggregate.reindex(times)
    if aggregate.isna().any().any():
        raise ValueError('Timestamp has no valid observations; no invented targets')
    if len(times) < 3 or not np.all(np.diff(times) == 300):
        raise ValueError('Expected complete five-minute grid')
    aggregate = aggregate.reset_index()
    audit = {'rows': len(raw), 'vms': raw.vm_id.nunique(), 'timestamps': len(times),
             'first_source_second': int(times[0]), 'last_source_second': int(times[-1]),
             'cadence_seconds': 300, 'invalid_avg_cpu_rows': int((~valid).sum()),
             'duplicate_pairs': 0, 'metadata_rows': len(meta),
             'missing_by_column': raw.isna().sum().to_dict(),
             'invalid_min_max_rows': int((~np.isfinite(raw.min_cpu) | ~np.isfinite(raw.max_cpu) | ~raw.min_cpu.between(0,100) | ~raw.max_cpu.between(0,100) | (raw.min_cpu > raw.avg_cpu) | (raw.avg_cpu > raw.max_cpu)).sum()),
             'observations_per_vm': raw.groupby('vm_id').size().describe().to_dict(),
             'missing_vm_grid_readings': int(len(times)*raw.vm_id.nunique()-len(raw))}
    return raw, meta, aggregate, audit


def forecast(y, horizon, model):
    y = np.asarray(y, dtype=float)
    if len(y) == 0 or not np.isfinite(y).all() or horizon <= 0:
        raise ValueError('Finite nonempty history and positive horizon required')
    if model == 'last':
        return np.repeat(y[-1], horizon)
    if model.startswith('mean'):
        return np.repeat(np.mean(y[-int(model[4:]):]), horizon)
    if model.startswith('ses'):
        alpha = {'ses02': .2, 'ses05': .5}[model]
        level = y[0]
        for value in y[1:]:
            level = alpha*value + (1-alpha)*level
        return np.repeat(level, horizon)
    if model == 'damped':
        recent = y[-min(12,len(y)):]
        slope = np.polyfit(np.arange(len(recent)), recent, 1)[0] if len(recent)>1 else 0.
        return y[-1] + slope*np.cumsum(.8**np.arange(1,horizon+1))
    if model == 'ar_ridge':
        lag = 6
        if len(y) < 2*lag+2 or np.std(y)<1e-12:
            return np.repeat(y[-1], horizon)
        center, scale = y.mean(), y.std()
        z = (y-center)/scale
        x = np.array([z[i-lag:i][::-1] for i in range(lag,len(z))])
        x = np.column_stack([np.ones(len(x)),x])
        penalty = np.eye(lag+1)*10.; penalty[0,0] = 0
        beta = np.linalg.solve(x.T@x+penalty, x.T@z[lag:])
        hist = list(z)
        for _ in range(horizon):
            hist.append(float(np.r_[1.,hist[-lag:][::-1]]@beta))
        return np.array(hist[-horizon:])*scale+center
    raise ValueError(f'Unknown model: {model}')


def bounded_forecast(y,horizon,model,target):
    pred = forecast(y,horizon,model)
    # Nonnegative counts; no constant 10,000 cap or hard-coded population.
    return np.maximum(pred,0) if target == TARGETS[0] else np.clip(pred,0,100)


def scores(actual,pred):
    actual, pred = np.asarray(actual), np.asarray(pred)
    if actual.shape != pred.shape or not np.isfinite(actual).all() or not np.isfinite(pred).all():
        raise ValueError('Metric inputs must be finite and equally shaped')
    e = pred-actual
    return {'mae':float(np.mean(np.abs(e))), 'rmse':float(np.sqrt(np.mean(e**2))),
            'bias':float(np.mean(e))}


def select_models(train,horizon=24):
    n = len(train)
    # Three nonoverlapping validation horizons when history permits. Adapt to
    # small smoke inputs without hard-coded release timestamps/population.
    minimum = max(8, horizon//2)
    origins = [n-3*horizon,n-2*horizon,n-horizon]
    origins = [o for o in origins if o>=minimum]
    if not origins:
        raise ValueError('Insufficient training history for a temporal fold')
    rows, records, selections, margins = [], [], {}, {}
    for target in TARGETS:
        for model in CANDIDATES:
            for origin in origins:
                y = train[target].iloc[:origin].to_numpy()
                actual = train[target].iloc[origin:origin+horizon].to_numpy()
                pred = bounded_forecast(y,horizon,model,target)
                rows.append({'target':target,'model':model,'origin':origin,**scores(actual,pred)})
                for k,(a,p) in enumerate(zip(actual,pred)):
                    records.append({'target':target,'model':model,'origin':origin,'step':k+1,
                                    'actual':float(a),'prediction':float(p),
                                    'peak_threshold':float(np.quantile(y,.9))})
        table = pd.DataFrame(rows)
        summary = table[table.target==target].groupby('model').mae.mean()
        # On numerical ties choose the simpler candidate in declaration order.
        best = summary.min()
        selected = next(m for m in CANDIDATES if summary[m]<=best+1e-12)
        selections[target] = selected
        selected_records = [r for r in records if r['target']==target and r['model']==selected]
        residuals = [r['actual']-r['prediction'] for r in selected_records]
        margins[target] = float(max(0,np.quantile(residuals,.95)))
    return selections,pd.DataFrame(rows),pd.DataFrame(records),margins,origins


def validate_predictions(pred,timestamps):
    if list(pred.columns)!=['timestamp',*TARGETS] or len(pred)!=len(timestamps):
        raise ValueError('Prediction schema/row count mismatch')
    if pred.timestamp.duplicated().any() or not np.array_equal(pred.timestamp.to_numpy(),timestamps):
        raise ValueError('Prediction timestamps mismatch')
    if not np.isfinite(pred.to_numpy(dtype=float)).all():
        raise ValueError('Nonfinite predictions')
    if (pred.active_vm_count<0).any() or not pred.aggregate_cpu_utilization.between(0,100).all():
        raise ValueError('Predictions outside bounds')


def training_analysis(raw,meta,train):
    subset = raw[(raw.timestamp<=train.timestamp.iloc[-1]) & np.isfinite(raw.avg_cpu) & raw.avg_cpu.between(0,100)].copy()
    # Metadata used for retrospective segmentation ONLY, never model features.
    merged = subset.merge(meta[['vm_id','vm_category','virtual_core_bucket']],on='vm_id',validate='many_to_one')
    segment = merged.groupby('vm_category').agg(vms=('vm_id','nunique'),mean_cpu=('avg_cpu','mean'),p95_cpu=('avg_cpu',lambda x:x.quantile(.95)),mean_interval_max=('max_cpu','mean'))
    segment['vm_share'] = segment.vms/segment.vms.sum()
    segment['share_of_summed_mean_cpu'] = merged.groupby('vm_category').avg_cpu.sum()/merged.avg_cpu.sum()
    stats = {'cpu_quantiles':subset.avg_cpu.quantile([0,.1,.5,.9,.95,.99,1]).to_dict(),
             'zero_cpu_fraction':float((subset.avg_cpu==0).mean()),
             'over_80_cpu_fraction':float((subset.avg_cpu>80).mean()),
             'active_count_min':int(train.active_vm_count.min()),'active_count_max':int(train.active_vm_count.max()),
             'aggregate_cpu_mean':float(train.aggregate_cpu_utilization.mean()),
             'aggregate_cpu_std':float(train.aggregate_cpu_utilization.std()),
             'aggregate_cpu_min':float(train.aggregate_cpu_utilization.min()),
             'aggregate_cpu_max':float(train.aggregate_cpu_utilization.max()),
             'aggregate_lag1_autocorrelation':float(train.aggregate_cpu_utilization.autocorr()),
             'observed_population_additions':int(subset.groupby('vm_id').timestamp.min().gt(subset.timestamp.min()).sum()),
             'observed_population_departures':int(subset.groupby('vm_id').timestamp.max().lt(subset.timestamp.max()).sum())}
    # Lifetime fields may describe the broader source trace and are not forecast features.
    for col in ['timestamp_vm_created','timestamp_vm_deleted']:
        if col in meta:
            stats[col+'_summary'] = pd.to_numeric(meta[col],errors='coerce').describe().to_dict()
    cores = merged.groupby('virtual_core_bucket').agg(vms=('vm_id','nunique'),mean_cpu=('avg_cpu','mean'))
    seg_ts = merged.groupby(['timestamp','vm_category']).avg_cpu.mean().unstack()
    return stats,segment,cores,seg_ts


def make_figures(raw,train,holdout,pred,records,segment,seg_ts,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'figure.dpi':140,'axes.spines.top':False,'axes.spines.right':False})
    out.mkdir(exist_ok=True,parents=True)
    fig,axes=plt.subplots(2,1,figsize=(9,5),sharex=True)
    for ax,t in zip(axes,TARGETS):
        ax.plot(np.arange(len(train)),train[t],label='Training',lw=1.3)
        ax.plot(np.arange(len(train),len(train)+len(pred)),holdout[t],label='Held-out actual',lw=1.3)
        ax.plot(np.arange(len(train),len(train)+len(pred)),pred[t],label='Frozen forecast',ls='--')
        ax.axvline(len(train)-.5,color='gray',ls=':');ax.set_ylabel(t.replace('_',' '));ax.legend(fontsize=7)
    axes[-1].set_xlabel('Five-minute observation index (not calendar time)')
    fig.tight_layout();fig.savefig(out/'forecast.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,3))
    raw.loc[(raw.timestamp<=train.timestamp.iloc[-1]) & np.isfinite(raw.avg_cpu) & raw.avg_cpu.between(0,100),'avg_cpu'].hist(bins=50,ax=axes[0]);axes[0].set_yscale('log');axes[0].set_xlabel('VM average CPU (%)');axes[0].set_ylabel('Readings (log)')
    segment.mean_cpu.plot.bar(ax=axes[1],rot=15);axes[1].set_ylabel('Training mean CPU (%)')
    fig.tight_layout();fig.savefig(out/'workload.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,2.8));seg_ts.reset_index(drop=True).plot(ax=ax);ax.set_xlabel('Training five-minute index');ax.set_ylabel('Segment mean CPU (%)');fig.tight_layout();fig.savefig(out/'segments.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,2.8));cpu=records[records.target==TARGETS[1]];cpu.assign(absolute_error=np.abs(cpu.actual-cpu.prediction)).groupby('model').absolute_error.mean().sort_values().plot.bar(ax=ax,rot=25);ax.set_ylabel('Validation CPU MAE (percentage points)');fig.tight_layout();fig.savefig(out/'validation.png');plt.close(fig)


def run(data,output,artifacts,horizon=24,check=True,evaluate=True):
    data,output,artifacts=Path(data),Path(output),Path(artifacts)
    artifacts.mkdir(parents=True,exist_ok=True)
    checksums=verify_checksums(data) if check else {'status':'explicitly skipped (smoke/test only)'}
    raw,meta,aggregate,audit=load_data(data)
    if len(aggregate)<=horizon:
        raise ValueError('Not enough history')
    if check and (len(aggregate)!=134 or horizon!=24 or raw.vm_id.nunique()!=10000 or len(raw)!=1340000):
        raise ValueError('Official release requires 134 timestamps, 10,000 VMs, 1,340,000 rows and horizon 24')
    train=aggregate.iloc[:-horizon].copy()
    # No held-out labels are passed to selection, analysis or forecasting.
    selected,folds,records,margins,origins=select_models(train,horizon)
    manifest={'selected':selected,'candidate_order':CANDIDATES,'selection_metric':'mean validation MAE; simpler candidate breaks ties',
              'origins':origins,'horizon':horizon,'training_points':len(train),
              'last_training_timestamp':int(train.timestamp.iloc[-1]),'one_sided_residual_p95_margin':margins,
              'upstream_starter_commit':'6e5ac0e4d2f1d81fab0903ce0a974037745db35b','checksums':checksums}
    # Persist frozen model selection BEFORE evaluating final labels.
    (artifacts/'selection.json').write_text(json.dumps(manifest,indent=2))
    pred=pd.DataFrame({'timestamp':aggregate.timestamp.iloc[-horizon:].to_numpy()})
    for t in TARGETS:
        pred[t]=bounded_forecast(train[t],horizon,selected[t],t)
    validate_predictions(pred,aggregate.timestamp.iloc[-horizon:].to_numpy())
    output.parent.mkdir(parents=True,exist_ok=True);pred.to_csv(output,index=False)
    audit['training_points']=len(train)
    stats,segment,cores,seg_ts=training_analysis(raw,meta,train)
    for name,obj in [('audit',audit),('training_analysis',stats)]:
        (artifacts/f'{name}.json').write_text(json.dumps(obj,indent=2))
    segment.to_csv(artifacts/'segments.csv');cores.to_csv(artifacts/'core_segments.csv')
    folds.to_csv(artifacts/'validation_folds.csv',index=False);records.to_csv(artifacts/'validation_predictions.csv',index=False)
    train.to_csv(artifacts/'training_targets.csv',index=False)
    if evaluate:
        holdout=aggregate.iloc[-horizon:]
        results=[]
        for t in TARGETS:
            threshold=float(train[t].quantile(.9));peak=holdout[t].to_numpy()>=threshold
            baseline=bounded_forecast(train[t],horizon,'last',t)
            for name,p in [('selected',pred[t].to_numpy()),('last_baseline',baseline)]:
                result={'target':t,'model':name,**scores(holdout[t],p),
                        'peak_threshold_training_p90':threshold,'peak_points':int(peak.sum()),
                        'peak_mae':float(np.mean(np.abs(holdout[t].to_numpy()[peak]-p[peak]))) if peak.any() else None,
                        'max_underprediction':float(max(0,np.max(holdout[t].to_numpy()-p))),
                        'buffer_exceedance_fraction':float(np.mean(holdout[t].to_numpy()>p+margins[t]))}
                baseline_mae=scores(holdout[t],baseline)['mae']
                result['mae_relative_to_persistence']=result['mae']/baseline_mae if baseline_mae>0 else None
                result['mae_improvement_percent']=100*(1-result['mae']/baseline_mae) if baseline_mae>0 else None
                result['relative_metric_note']='defined' if baseline_mae>0 else 'undefined: persistence MAE is zero; compare absolute errors'
                results.append(result)
        pd.DataFrame(results).to_csv(artifacts/'holdout_metrics.csv',index=False)
        holdout.to_csv(artifacts/'holdout_actuals.csv',index=False)
        make_figures(raw,train,holdout,pred,records,segment,seg_ts,artifacts/'figures')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',default='challenge_data');parser.add_argument('--output',default='predictions.csv')
    parser.add_argument('--artifacts',default='reports/artifacts');parser.add_argument('--horizon',type=int,default=24)
    parser.add_argument('--skip-checksums',action='store_true');parser.add_argument('--no-evaluate',action='store_true')
    args=parser.parse_args()
    print(json.dumps(run(args.data,args.output,args.artifacts,args.horizon,not args.skip_checksums,not args.no_evaluate),indent=2))

if __name__=='__main__':
    main()
