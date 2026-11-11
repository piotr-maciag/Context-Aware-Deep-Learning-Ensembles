"""Audit saved Town Neck results and produce context/aggregate reports."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef

from Source.bs_town_neck_training import current_split_provenance
from Source.ensemble import (average_probabilities, hard_voting_predictions,
    weighted_average_probabilities, class_weighted_average_probabilities)

BACKBONES = {
    "small_cnn": "Small CNN", "efficientnetv2_s": "EfficientNetV2-S",
    "vit_tiny": "ViT-Tiny", "swin_tiny": "Swin-Tiny", "convnext_tiny": "ConvNeXt-Tiny",
}
ENSEMBLES = {
    "hard_voting": "Hard voting", "probability_voting": "Probability voting",
    "weighted_probability_per_model": "Globally weighted probability voting",
    "weighted_probability_per_model_class": "Model-and-class weighted probability voting",
}
METRICS = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "mcc"]


def _close(actual, expected, context, atol=1e-10):
    if not np.allclose(actual, expected, rtol=0, atol=atol):
        raise ValueError(f"Saved/recomputed results disagree: {context}")


def _metrics(truth, prediction, codes):
    p,r,f,_ = precision_recall_fscore_support(truth, prediction, labels=codes,
                                            average="macro", zero_division=0)
    return dict(zip(METRICS,[accuracy_score(truth,prediction), p,r,f,
                             matthews_corrcoef(truth,prediction)]))


def _predictions(path, test):
    frame = pd.read_csv(path, dtype={"date":str,"cell_id":str,"class":str,"true_class_code":str})
    if frame.duplicated(["date","cell_id"]).any():
        raise ValueError(f"Duplicate prediction records: {path}")
    for column in ["date","cell_id","class","split"]:
        if not frame[column].astype(str).equals(test[column].astype(str)):
            raise ValueError(f"Prediction {column}/order differs from test split: {path}")
    if not frame.true_class_code.equals(test["class"]):
        raise ValueError(f"Ground truth differs from test split: {path}")
    return frame


def load_town_neck_report(project_root):
    root = Path(project_root)
    split_root = root / "Data/Preprocessed/bs_town_neck"
    provenance = current_split_provenance(split_root)
    test = pd.read_csv(split_root / "bs_test_split.csv",dtype=str)
    codes = sorted(test["class"].astype(int).unique())
    if codes != [1,2,3,4,5,6]:
        raise ValueError("Expected all six Town Neck classes")
    truth = test["class"].astype(int).to_numpy()
    rows, runs, probabilities, predictions, metadata = [], [], {}, {}, {}
    for name,label in BACKBONES.items():
        directory = root / "Results" / f"bs_town_neck_{name}" / "RGB_only"
        if (directory / "experiment_in_progress.json").exists():
            raise ValueError(f"Experiment incomplete: {directory}")
        m = json.loads((directory / "inference_metadata.json").read_text())
        if (m["split_provenance"] != provenance or m["feature_suffixes"] != ["r","g","b"]
            or m.get("grid_rows") != provenance["grid_rows"]
            or m.get("grid_columns") != provenance["grid_columns"]):
            raise ValueError(f"Stale or incompatible model provenance: {name}")
        class_order = [int(m["idx_to_class_code"][str(i)]) for i in range(6)]
        if class_order != codes or m["class_to_idx"] != {str(c):i for i,c in enumerate(codes)}:
            raise ValueError(f"Class order mismatch: {name}")
        config = json.loads((directory / "best_config.json").read_text())
        if config != m["best_config"] or int(config["context_radius"]) != m["context_radius"]:
            raise ValueError(f"Saved best configuration mismatch: {name}")
        grid = pd.read_csv(directory / "grid_search_summary.csv")
        if (len(grid) != 16 or grid.run_index.duplicated().any()
            or set(grid.run_index) != set(range(1,17))
            or grid.duplicated(["context_radius","learning_rate","weight_decay"]).any()
            or set(grid.context_radius) != {0,1,2,3}
            or not grid.groupby("context_radius").size().eq(4).all()
            or not np.isfinite(grid.validation_macro_f1).all()
            or not grid.validation_macro_f1.between(0,1).all()):
            raise ValueError(f"Incomplete or invalid grid-search summary: {name}")
        for _,group in grid.groupby("context_radius"):
            expected = {(lr,wd) for lr in [1e-4,3e-5] for wd in [1e-4,1e-3]}
            if set(zip(group.learning_rate,group.weight_decay)) != expected:
                raise ValueError(f"Unexpected parameter combinations: {name}")
        winner = grid.sort_values(["validation_macro_f1","run_index"],ascending=[False,True]).iloc[0]
        for key in ["context_radius","learning_rate","weight_decay", "dropout"]:
            if key in config:
                _close(float(config[key]),float(winner[key]),f"{name} selected {key}")
        saved_validation = pd.read_csv(directory / "best_validation_classification_report.csv",index_col=0)
        _close(saved_validation.loc["macro avg","f1-score"],winner.validation_macro_f1,
               f"{name} validation winner")
        frame = _predictions(directory / "test_predictions.csv",test)
        prob = frame[[f"probability_class_{code}" for code in codes]].to_numpy(float)
        if not np.isfinite(prob).all() or (prob < 0).any() or (prob > 1).any():
            raise ValueError(f"Invalid probabilities: {name}")
        _close(prob.sum(axis=1),1,f"{name} probability sums",atol=1e-6)
        prediction = frame.predicted_class_code.to_numpy(int)
        if not np.array_equal(prediction,np.asarray(codes)[prob.argmax(axis=1)]):
            raise ValueError(f"Predictions disagree with probabilities: {name}")
        metrics = _metrics(truth,prediction,codes)
        saved = pd.read_csv(directory / "test_aggregate_metrics.csv").iloc[0]
        _close([metrics[k] for k in METRICS],saved[METRICS].to_numpy(float),name)
        rows.append({"method_id":name,"Method":label,"Context":str(m["context_radius"]),
                     "Validation F1":float(winner.validation_macro_f1),**metrics})
        runs.append(grid.assign(model=name,Method=label))
        probabilities[name], predictions[name], metadata[name] = prob,prediction,m
    ensemble_dir = root / "Results/bs_town_neck_ensemble/RGB_only"
    if (ensemble_dir / "experiment_in_progress.json").exists():
        raise ValueError("Ensemble evaluation is incomplete")
    definition = json.loads((ensemble_dir / "ensemble_definition.json").read_text())
    if definition["split_provenance"] != provenance or definition["feature_suffixes"] != ["r","g","b"]:
        raise ValueError("Stale ensemble split/channel provenance")
    component_order = ["efficientnetv2_s","vit_tiny","swin_tiny","convnext_tiny"]
    if set(definition["models"]) != set(component_order):
        raise ValueError("Expected the four advanced backbones in the ensemble")
    for name in component_order:
        digest = sha256(json.dumps(metadata[name],sort_keys=True).encode()).hexdigest()
        if digest != definition["deployment_model_metadata_hashes"][name]:
            raise ValueError(f"Ensemble used a different model artifact: {name}")
    component_probs = {name:probabilities[name] for name in component_order}
    hard_prediction,hard_probability = hard_voting_predictions(component_probs)
    class_weights = {name:np.array([definition["weights_by_model_and_class"][name][str(c)]
                                   for c in codes]) for name in component_order}
    combined = {
        "hard_voting":(hard_prediction,hard_probability),
        "probability_voting":(None,average_probabilities(component_probs)),
        "weighted_probability_per_model":(None,weighted_average_probabilities(
            component_probs,definition["global_model_weights"])),
        "weighted_probability_per_model_class":(None,class_weighted_average_probabilities(
            component_probs,class_weights)),
    }
    frame = _predictions(ensemble_dir / "test_predictions.csv",test)
    comparison = pd.read_csv(ensemble_dir / "test_model_comparison.csv").set_index("model")
    search = pd.read_csv(ensemble_dir / "validation_ensemble_search.csv")
    selected = max(ENSEMBLES,key=lambda method:float(search.set_index("method").loc[method,"macro_f1"]))
    if selected != definition["validation_selected_method"]:
        raise ValueError("Ensemble validation selection mismatch")
    for name in component_order:
        if not np.array_equal(frame[f"{name}_predicted_class_code"].to_numpy(int),predictions[name]):
            raise ValueError(f"Ensemble and standalone predictions disagree: {name}")
        expected = _metrics(truth,predictions[name],codes)
        _close([expected[k] for k in METRICS],comparison.loc[name,METRICS].to_numpy(float),name)
    vector = '('+','.join(str(metadata[name]["context_radius"]) for name in component_order)+')'
    for method,label in ENSEMBLES.items():
        prediction,prob = combined[method]
        if prediction is None:
            prediction = prob.argmax(axis=1)
        prediction = np.asarray(codes)[prediction]
        if not np.array_equal(prediction,frame[f"{method}_predicted_class_code"].to_numpy(int)):
            raise ValueError(f"Saved ensemble predictions do not reproduce: {method}")
        _close(prob,frame[[f"{method}_probability_class_{c}" for c in codes]].to_numpy(float),method,atol=1e-6)
        metrics = _metrics(truth,prediction,codes)
        _close([metrics[k] for k in METRICS],comparison.loc[f"ensemble_{method}",METRICS].to_numpy(float),method)
        rows.append({"method_id":method,"Method":label,"Context":vector,
                     "Validation F1":float(search.set_index("method").loc[method,"macro_f1"]),**metrics})
    run_frame = pd.concat(runs,ignore_index=True)
    context_summary = run_frame.groupby(["model","Method","context_radius"],sort=False).agg(
        mean=("validation_macro_f1","mean"),sd=("validation_macro_f1","std"),
        runs=("validation_macro_f1","size")).reset_index()
    return pd.DataFrame(rows),context_summary,selected


def plot_context_f1(summary):
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots(figsize=(7.5,4.8))
    x = np.arange(len(BACKBONES)); width=.19
    context_hatches = ['', '//', 'xx', '..']
    colors = ['#377eb8','#e69f00','#009e73','#cc6677']
    for radius,hatch in enumerate(context_hatches):
        data = summary.loc[summary.context_radius.eq(radius)].set_index('model').reindex(BACKBONES)
        ax.bar(x+(radius-1.5)*width,data['mean'],width,yerr=data.sd,capsize=3,
               label=f'Radius {radius}',color=colors[radius],hatch=hatch,
               edgecolor='#242424',linewidth=.65,
               error_kw={'elinewidth':1.1,'capthick':1.1,'ecolor':'#202020'})
    ax.set_xticks(x,['Small\nCNN','EfficientNet\nV2-S','ViT-Tiny','Swin-Tiny','ConvNeXt\nTiny'],fontsize=12)
    ax.set_ylabel('Validation macro-F1',fontsize=13)
    ax.set_xlabel('Backbone model',fontsize=13)
    ax.tick_params(axis='y',labelsize=11)
    ax.set_ylim(0,1)
    ax.set_title('Town Neck Beach: mean and SD\nacross four grid-search configurations',fontsize=13,pad=10)
    ax.legend(ncol=4,fontsize=10,loc='upper left',frameon=False)
    ax.grid(axis='y',color='#d5d5d5',linewidth=.8);ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    return fig


def aggregate_latex(metrics):
    lines = [r'\begin{table}[!t]',
        r'\caption{Aggregated test-set metrics for Town Neck Beach after fine-tuning the RGB-only Svalbard models. Each backbone uses its validation-selected configuration. Precision, recall, and F1-score are macro averages over six classes. Ensemble radius vectors follow EfficientNetV2-S, ViT-Tiny, Swin-Tiny, and ConvNeXt-Tiny. The best unrounded values are shaded in grey.}',
        r'\label{tab:town-neck-aggregate-metrics}',r'\centering',r'\footnotesize',
        r'\setlength{\tabcolsep}{4.0pt}',r'\renewcommand{\arraystretch}{1.05}',
        r'\begin{tabular}{lrrrrr}',r'\hline',
        r'\textbf{Method} & \textbf{Acc.} & \textbf{Prec.} & \textbf{Rec.} & \textbf{F1-score} & \textbf{MCC} \\',r'\hline']
    maximum = metrics[METRICS].max()
    for row in metrics.to_dict('records'):
        label = row['Method']
        if row['method_id']=='weighted_probability_per_model':
            label = r'Globally weighted\\probability voting'
        elif row['method_id']=='weighted_probability_per_model_class':
            label = r'Model-and-class\\weighted probability\\voting'
        context = (r'$\mathbf{r}='+row['Context']+'$' if row['method_id'] in ENSEMBLES
                   else '$r='+row['Context']+'$')
        name = r'\shortstack[l]{'+label+r'\\RGB only; '+context+'}'
        values = [(r'\cellcolor{gray!25}' if np.isclose(row[k],maximum[k],rtol=0,atol=1e-12) else '')
                  +f'{row[k]:.3f}' for k in METRICS]
        lines += [name+' & '+' & '.join(values)+r' \\',r'\hline']
    lines += [r'\end{tabular}',r'\end{table}']
    return '\n'.join(lines)
