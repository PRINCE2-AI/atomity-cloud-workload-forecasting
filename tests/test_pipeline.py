from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from cwfc.pipeline import (forecast, bounded_forecast, select_models, load_data,
                           validate_predictions, run, verify_checksums, TARGETS)


def fixture_data(path, n=64, vms=3):
    path.mkdir(parents=True, exist_ok=True)
    rows=[]
    for i in range(n):
        for vm in range(vms):
            avg=15+vm+np.sin(i/5)
            rows.append([300*(i+1),f'vm{vm}',avg-1,avg+1,avg])
    pd.DataFrame(rows,columns=['timestamp','vm_id','min_cpu','max_cpu','avg_cpu']).to_csv(path/'vm_cpu_readings.csv.gz',index=False)
    pd.DataFrame({'vm_id':[f'vm{i}' for i in range(vms)],'vm_category':['Interactive']*vms,'virtual_core_bucket':['2']*vms}).to_csv(path/'vm_metadata.csv.gz',index=False)


@pytest.mark.parametrize('model',['last','mean6','mean12','mean24','ses02','ses05','damped','ar_ridge'])
def test_constant_series(model):
    assert np.allclose(forecast(np.ones(40)*7,24,model),7)


def test_boundaries():
    assert (bounded_forecast(np.array([99,100,100]),24,'damped',TARGETS[1])<=100).all()
    with pytest.raises(ValueError): forecast([],24,'last')


def test_selection_training_only():
    frame=pd.DataFrame({'active_vm_count':np.ones(110)*3,'aggregate_cpu_utilization':20+np.sin(np.arange(110)/9)})
    selection,folds,records,_,origins=select_models(frame)
    assert origins==[38,62,86]
    assert selection[TARGETS[0]]=='last'
    assert len(records)==2*8*3*24
    assert (folds.origin+24<=110).all()


def test_smoke_and_holdout_independence(tmp_path):
    data=tmp_path/'data';fixture_data(data)
    first=run(data,tmp_path/'p1.csv',tmp_path/'a1',horizon=8,check=False,evaluate=False)
    raw=pd.read_csv(data/'vm_cpu_readings.csv.gz')
    raw.loc[raw.timestamp>56*300,['avg_cpu','min_cpu','max_cpu']]=[90,89,91]
    raw.to_csv(data/'vm_cpu_readings.csv.gz',index=False)
    second=run(data,tmp_path/'p2.csv',tmp_path/'a2',horizon=8,check=False,evaluate=False)
    assert first['selected']==second['selected']
    assert (tmp_path/'p1.csv').read_bytes()==(tmp_path/'p2.csv').read_bytes()
    pred=pd.read_csv(tmp_path/'p1.csv')
    assert len(pred)==8 and (pred.active_vm_count==3).all()


def test_duplicate_rejection(tmp_path):
    fixture_data(tmp_path)
    raw=pd.read_csv(tmp_path/'vm_cpu_readings.csv.gz');pd.concat([raw,raw.iloc[:1]]).to_csv(tmp_path/'vm_cpu_readings.csv.gz',index=False)
    with pytest.raises(ValueError,match='Duplicate'):load_data(tmp_path)


def test_invalid_reading_excluded_from_both_targets(tmp_path):
    fixture_data(tmp_path)
    raw=pd.read_csv(tmp_path/'vm_cpu_readings.csv.gz');raw.loc[0,'avg_cpu']=np.nan;raw.to_csv(tmp_path/'vm_cpu_readings.csv.gz',index=False)
    _,_,agg,audit=load_data(tmp_path)
    assert agg.active_vm_count.iloc[0]==2 and audit['invalid_avg_cpu_rows']==1


def test_gap_rejection(tmp_path):
    fixture_data(tmp_path)
    raw=pd.read_csv(tmp_path/'vm_cpu_readings.csv.gz');raw[raw.timestamp!=600].to_csv(tmp_path/'vm_cpu_readings.csv.gz',index=False)
    with pytest.raises(ValueError,match='grid'):load_data(tmp_path)


def test_output_contract():
    frame=pd.DataFrame({'timestamp':[300,600],TARGETS[0]:[3.,3.],TARGETS[1]:[20.,21.]})
    validate_predictions(frame,np.array([300,600]))
    frame.loc[1,'timestamp']=300
    with pytest.raises(ValueError):validate_predictions(frame,np.array([300,600]))


def test_checksum_failure(tmp_path):
    (tmp_path/'test').write_text('x');(tmp_path/'SHA256SUMS').write_text('0'*64+'  test\n')
    with pytest.raises(ValueError,match='Checksum'):verify_checksums(tmp_path)


def test_default_evaluation_path(tmp_path):
    data=tmp_path/"data";fixture_data(data)
    run(data,tmp_path/"pred.csv",tmp_path/"artifacts",horizon=8,check=False,evaluate=True)
    assert (tmp_path/"artifacts/figures/forecast.png").exists()
    metrics=pd.read_csv(tmp_path/"artifacts/holdout_metrics.csv")
    assert "mae_relative_to_persistence" in metrics


def test_zero_step_rejected():
    from cwfc.validation import rolling_origin_splits
    with pytest.raises(ValueError,match="step"):
        rolling_origin_splits(pd.DataFrame({"timestamp":[0,300,600,900]}),horizon=1,min_train_size=2,step=0)
