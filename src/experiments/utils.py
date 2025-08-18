import os
import json
import copy
import torch
import random
import pickle
import typing
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
from typing import Any, Literal, List, Dict, Optional, Tuple
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer

from src.sem.simulation.linear import COVARIATE_DIMENSION


Experiment = Literal[
    'linear_simulation',
    'nonlinear_simulation',
    'optical_device',
    'colored_mnist',
    'rotated_mnist'
]
Plot = Literal['png', 'pdf', 'ps', 'eps', 'svg']

FS_TICK: int = 18
FS_LABEL: int = 24
PLOT_DPI: int=1200
PAGE_WIDTH: float=6.75
PLOT_FORMAT: Plot='pdf'
HILIGHT_OURS: bool=False
RICE_AUGMENTATIONS: int=3
ARTIFACTS_DIRECTORY: str='artifacts'
RC_PARAMS: Dict[str, str | int | bool] = {
    # # Set LaTeX for rendering text.
    # # Uncomment this only if you have installed latex dependencies.
    # 'text.usetex': True,
    # 'font.family': 'serif',
    # 'font.serif': ['Computer Modern'],
    # 'text.latex.preamble': r'\usepackage{amsmath}',
    # # Set background and border settings
    # 'axes.facecolor': 'white',
    # 'axes.edgecolor': 'black',
    # 'axes.linewidth': 2,
    # 'xtick.color': 'black',
    # 'ytick.color': 'black',
}
TEX_MAPPER: Dict[str, str] = {
    'Data': r'Data',
    'ERM': r'ERM',
    'DA+ERM': r'DA+ERM',
    'DA+IVL-a': r'DA+IVL$_\alpha$',
    'DA+IVL-CV': r'DA+IVL$_\alpha^{_\text{CV}}$',
    'DA+IVL-LCV': r'DA+IVL$_\alpha^{_\text{LCV}}$',
    'DA+IVL-CC': r'DA+IVL$_\alpha^{_\text{CC}}$',
    'DA+IVL-Pi': r'DA+IVL$_\Pi$',
    'DA+IVL': r'DA+IVL',
    'DA+IV': r'DA+IV',
    'IRM': r'IRM',
    'ICP': r'ICP',
    'DRO': r'DRO',
    'RICE': r'RICE',
    'V-REx': r'V-REx',
    'MM-REx': r'MM-REx',
    'L1Janzing': r'$\ell_1$ Janzing `19',
    'L2Janzing': r'$\ell_2$ Janzing `19',
    'Kania&Wit': r'Kania, Wit `23',
}
ANNOTATE_BOX_PLOT: Dict[Experiment, Dict[str, Any]] = {
    'linear_simulation': {
        'title': 'Simulation Data',
    },
    'optical_device': {
        'title': 'Optical Device Data',
        # 'y_color': 'w',
    },
    'colored_mnist': {
        'title': 'Colored MNIST Data',
        'dummies': ['DA+IVL-CC', 'ICP', 'L1Janzing', 'L2Janzing', 'Kania&Wit'],
        # 'y_color': 'w',
    }
}
ANNOTATE_SWEEP_PLOT: Dict[str, Dict[str, Any]] = {
    'lambda': {
        'xlabel': r'$\lambda$',
        'xscale': 'linear',
        'hide_legend': True,
    },
    'alpha': {
        'xlabel': r'$\alpha$',
        'xscale': 'log',
        'vertical_plots': ['DA+IVL-CV', 'DA+IVL-LCV', 'DA+IVL-CC'],
        'trivial_solution': True,
        'legend_items': ['DA+IVL-CV', 'DA+IVL-LCV', 'DA+IVL-CC', 'DA+IVL-a'],
        # 'y_color': 'w',
        # 'legend_loc': (0.645, 0.425),
    },
    'gamma': {
        'xlabel': r'$\gamma$',
        'xscale': 'log',
        # 'y_color': 'w',
    }
}


def discretize(
        G: NDArray,
        n_bins: int=2,
        strategy: str='uniform'
    ):
    binner = KBinsDiscretizer(
        n_bins=n_bins, encode='ordinal', strategy=strategy
    )
    scaler = StandardScaler()
    G = binner.fit_transform(G)
    G = scaler.fit_transform(G).round(decimals=2)
    return G


def relative_error(W, What) -> float:
    sq_norm = lambda x: (x**2).sum()
    sq_error = sq_norm(What - W)
    relative_sq_error = sq_error / (sq_error + sq_norm(W))
    return relative_sq_error**0.5


def set_seed(seed: int=42):
    np.random.seed(seed)
    
    random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    logger.info(f'Random seed set as {seed}.')


def sweep_plot(
        x, y,
        xlabel: str,
        ylabel: Optional[str]='Retalive Error',
        xscale: Optional[Literal['linear', 'log']]='linear',
        vertical_plots: Optional[List]=[],
        trivial_solution: Optional[bool]=True,
        savefig: Optional[bool]=True,
        format: Optional[Plot]=PLOT_FORMAT,
        legend_items: Optional[List]=[],
        legend_loc: Optional[str | Tuple[float, float]]='best',
        y_color: Optional[bool]='k',
        hide_legend: Optional[bool]=False,
        hilight_ours: Optional[bool]=HILIGHT_OURS,
        bootstrapped: Optional[bool]=True,
    ):
    if bootstrapped:
        y = bootstrap(y)
    
    legend_items = [item for item in legend_items if item in y]

    # Define color palette (e.g., 'deep') and style (e.g., 'ticks')
    plt.rcParams.update(RC_PARAMS)
    sns.set_palette('deep')
    colors = sns.color_palette()
    fig = plt.figure()
    all_labels = []
    plot_handles = []
    for i, (method, errors) in enumerate(y.items()):

        # if method == 'DA+IVL-Pi' or method == 'DA+IVL':
        #     continue
        # if 'CV' in method or 'LCV' in method or 'CC' in method:
        #     continue
        
        mean = errors.mean(axis = 1)

        label = TEX_MAPPER.get(method, method)
        if method in vertical_plots:
            label = f'average {label.split("--")[-1]}'
        all_labels.append(label)
        if method in legend_items:
            legend_items[legend_items.index(method)] = label

        if method in vertical_plots:
            handle = plt.axvline(
                x=mean.mean(), color=colors[i], label=label, linestyle='--'
            )
        else:
            handle = plt.plot(x, mean, color=colors[i], label=label)[0]
        
        plot_handles.append(handle)
    
    if trivial_solution:
        label = fr'$0_{{{COVARIATE_DIMENSION}}}$'
        all_labels.append(label)
        if method in legend_items:
            legend_items[legend_items.index(method)] = label
        
        handle = plt.axhline(
            y = 0.5**0.5, color=colors[-1], label=label
        )
        plot_handles.append(handle)
        
    for i, (method, errors) in enumerate(y.items()):

        # if method == 'DA+IVL-Pi' or method == 'DA+IVL':
        #     continue
        # if 'CV' in method or 'LCV' in method or 'CC' in method:
        #     continue
        
        if method not in vertical_plots:
            low = np.percentile(errors, 2.5, axis=1)
            high = np.percentile(errors, 97.5, axis=1)
            plt.fill_between(x, low, high, color=colors[i], alpha = 0.2)
    
    plt.xlabel(xlabel, fontsize=FS_LABEL)
    plt.ylabel(ylabel, fontsize=FS_LABEL, color=y_color)
    plt.yticks(fontsize=FS_TICK, color=y_color)
    plt.xticks(fontsize=FS_TICK)
    plt.xlim([min(x), max(x)])
    maximum = 0.5**0.5
    padding = 0.05 * maximum
    plt.ylim([0.0 - padding, maximum + padding])
    plt.xscale(xscale)

    # Legend all items if None are specified
    if legend_items:
        labels = legend_items
    else:
        labels = all_labels
    handles = [plot_handles[all_labels.index(item)] for item in labels]

    if hilight_ours:
        for i, label in enumerate(labels):
            if label == TEX_MAPPER['DA+IVL-a']:
                continue
            elif 'IVL' in label or 'average' in label:
                bold = label
                bold = bold.replace(r'\alpha',r'{\boldsymbol{\alpha}}')
                bold = bold.replace(r'\Pi',r'{\boldsymbol{\Pi}}')
                bold = fr'\textbf{{{bold}}}'
                labels[i] = bold                

    if not hide_legend:
        plt.legend(
            handles=handles, labels=labels, fontsize=FS_TICK,
            loc=legend_loc, frameon=True, edgecolor='black', fancybox=False
        )

    plt.tight_layout()
    plt.show()
    if savefig:
        fname = ''.join(c for c in xlabel if c.isalnum()) + '_sweep'
        save(
            obj=fig,
            fname=fname,
            experiment='linear_simulation',
            format=format,
            dpi=PLOT_DPI
        )


def populate_dummy_data(
        data: Dict[str, Dict[str, NDArray]], dummies: List[str],
        scaler: Optional[float]=0.0
    ):
    dummies = [item for item in dummies if item in TEX_MAPPER]
    if dummies:
        data = copy.deepcopy(data)
        data_shape = list(list(data.values())[0].values())[0].shape
        dummy_data = {
            dummy: scaler * np.ones(data_shape) for dummy in dummies
        }
        data_with_dummies = {key: {} for key in data}
        for key in data:
            for method in TEX_MAPPER:
                if method in data[key]:
                    data_with_dummies[key][method] = data[key][method]
                elif method in dummies:
                    data_with_dummies[key][method] = dummy_data[method]
        return data_with_dummies
    else:
        return data


def box_plot(
        data: Dict[str, NDArray],
        fname: str,
        experiment: Experiment,
        title: Optional[str]='',
        xlabel: Optional[str]='Relative Error',
        ylabel: Optional[str]='Method',
        zlabel: Optional[str]='Augmentation',
        orient: Optional[Literal['h', 'v']]='h',
        savefig: Optional[bool]=True,
        format: Optional[Plot]=PLOT_FORMAT,
        annotate_best: Optional[bool]=True,
        dummies: Optional[List[str]]=[],
        y_color: Optional[bool]='k',
        hilight_ours: Optional[bool]=HILIGHT_OURS,
        bootstrapped: Optional[bool]=True,
    ):
    if bootstrapped:
        data = bootstrap(data)
    
    def prepare_data_for_plotting(
            data: Dict[str, Dict[str, NDArray]]
        ) -> pd.DataFrame:
        records = []
        minimum, maximum = float('inf'), float('-inf')
        for augmentation, methods in data.items():
            for method, values in methods.items():
                for value in values.flatten():
                    records.append({
                        zlabel: augmentation,
                        ylabel: TEX_MAPPER.get(method, method),
                        xlabel: value
                    })
                    if method not in dummies:
                        minimum = min(value, minimum)
                        maximum = max(value, maximum)
        df = pd.DataFrame.from_records(records)
        return df, minimum, maximum
    
    # check if data keys are subset of TEX_MAPPER keys
    # i.e., check if data keys only correspond to methods
    # if yes, then dont use zlabel as hue, else use zlabel.
    single_row = (
        set(data) <= set(TEX_MAPPER) or len(data) == 1
    )
    if single_row:
        zlabel = ylabel
        if len(data) > 1:
            data = {None : data}
    
    if 'error' in (xlabel.lower() + ylabel.lower()):
        data = populate_dummy_data(data, dummies, scaler=2.0)
    elif 'accuracy' in (xlabel.lower() + ylabel.lower()):
        data = populate_dummy_data(data, dummies, scaler=-1.0)
    else:
        raise ValueError(
            'Specify either `error` or `accuracy` in `xlabel` or `ylabel`.'
        )
    
    df, minimum, maximum = prepare_data_for_plotting(data)

    if annotate_best and single_row:
        average_scores = df.groupby(ylabel, sort=False).mean()[xlabel]
        if 'error' in (xlabel.lower() + ylabel.lower()):
            best_idx = average_scores.argmin()
        elif 'accuracy' in (xlabel.lower() + ylabel.lower()):
            best_idx = average_scores.argmax()
        else:
            raise ValueError(
                'Specify either `error` or `accuracy` in `xlabel` or `ylabel`.'
            )
    
    # Define color palette (e.g., 'deep') and style (e.g., 'ticks')
    plt.rcParams.update(RC_PARAMS)
    sns.set_palette('deep')
    fig = plt.figure()

    if orient == 'v':
        xlabel, ylabel = ylabel, xlabel

    ax = sns.boxplot(
        x=xlabel, y=ylabel,
        hue=zlabel,
        data=df,
        palette='deep',
        orient=orient,
        showmeans=True,
        meanprops={
            'markerfacecolor': 'white',
            'markeredgecolor': 'black'
            },
        flierprops={'marker': 'x'}
    )

    spread = maximum - minimum
    padding = 0.05 * spread
    if orient == 'h':
        plt.xlim([minimum - padding, maximum + padding])
    else:
        plt.ylim([minimum - padding, maximum + padding])
    
    if title:
        plt.title(title, fontsize=FS_LABEL)
    plt.ylabel('', fontsize=FS_LABEL, color=y_color)
    plt.xlabel(xlabel, fontsize=FS_LABEL)
    plt.xticks(fontsize=FS_TICK)
    plt.yticks(fontsize=FS_TICK, color=y_color)

    if dummies and single_row:
        method_ordered_list = list(list(data.values())[0].keys())
        for dummy in dummies:
            dummy_idx = method_ordered_list.index(dummy)
            if orient == 'h':
                plt.axhline(dummy_idx, color='r', linestyle='--', alpha=0.333)
            else:
                plt.axhline(dummy_idx, color='r', linestyle='--', alpha=0.333)

    if annotate_best and single_row:
        padding = 0.45
        if orient == 'v':
            plt.axvspan(best_idx-padding,best_idx+padding, color='r', alpha=0.1)
        else:
            plt.axhspan(best_idx-0.45,best_idx+0.45, color='r', alpha=0.1)
    
    def bold_tick(tick):
        tick.set_fontweight('bold')
        bold = tick.get_text()
        bold = bold.replace(r'\alpha',r'{\boldsymbol{\alpha}}')
        bold = bold.replace(r'\Pi',r'{\boldsymbol{\Pi}}')
        bold = fr'\textbf{{{bold}}}'
        tick.set_text(bold)
        return tick

    if hilight_ours:
        if orient == 'h':
            new_ticks = []
            for tick in ax.get_yticklabels():
                if 'IVL' in tick.get_text():
                    tick = bold_tick(tick)
                new_ticks.append(tick)
            ax.set_yticklabels(new_ticks)
        else:
            new_ticks = []
            for tick in ax.get_xticklabels():
                if 'IVL' in tick.get_text():
                    tick = bold_tick(tick)
                new_ticks.append(tick)
            ax.set_xticklabels(new_ticks)

    plt.tight_layout()
    plt.show()
    
    if savefig:
        save(
            obj=fig,
            fname=fname,
            experiment=experiment,
            format=format,
            dpi=PLOT_DPI
        )


def tex_table(
        data: Dict,
        label: str,
        caption: str,
        highlight: Literal['min', 'max']='min',
        decimals: int=3,
        hilight_ours: Optional[bool]=HILIGHT_OURS,
        bootstrapped: Optional[bool]=True,
    ):
    if bootstrapped:
        data = bootstrap(data)
    # check if data keys are subset of TEX_MAPPER keys
    # i.e., check if data keys only correspond to methods
    # if yes, then we need to construct a single row table
    single_row = set(data) <= set(TEX_MAPPER)
    if single_row:
        results = ([
            np.round((np.mean(v), np.std(v)), decimals) for v in data.values()
        ])
        if highlight == 'min':
            ordered = sorted(results, key=lambda x: (x[0], x[1]))
        elif highlight == 'max':
            ordered = sorted(results, key=lambda x: (-x[0], x[1]))
        best = ordered[0]
        second = ordered[1]
        column_names = [TEX_MAPPER.get(k, k) for k in data]
    else:
        row_names = list(data.keys())
        results = {}
        best = {}
        second = {}
        for row in row_names:
            columns = {
                col: data[row][col] for col in TEX_MAPPER.keys() if col in data[row]
            }
            results[row] = ([
                np.round((np.mean(v), np.std(v)), decimals) for v in columns.values()
            ])
            if highlight == 'min':
                ordered = sorted(results[row], key=lambda x: (x[0], x[1]))
            elif highlight == 'max':
                ordered = sorted(results[row], key=lambda x: (-x[0], x[1]))
            best[row] = ordered[0]
            second[row] = ordered[1]
        column_names = [TEX_MAPPER.get(k, k) for k in data[row_names[0]]]
    
    backreturn = '\\\\\n' + ' '*8

    num_columns = len(column_names) + int(not single_row)
    columns_preamble = ' '.join(['c']*num_columns)

    if hilight_ours:
        bold_column_names = []
        for name in column_names:
            if 'IVL' in name:
                bold = name
                bold = bold.replace(r'\alpha',r'{\boldsymbol{\alpha}}')
                bold = bold.replace(r'\Pi',r'{\boldsymbol{\Pi}}')
                bold = fr'\textbf{{{bold}}}'
                name = bold
            bold_column_names.append(name)
        column_names = bold_column_names

    columns = ' & '.join(column_names)
    if not single_row:
        columns = ' & ' + columns
    
    def row_content(row_data, best, second):
        if highlight == 'min':
            row = ' & '.join([
                ( f'${mean:.3f} \\pm {std:.3f}$' ) if mean > second
                else ( f'$\\mathit{{ {mean:.3f} \\pm {std:.3f} }}$' ) if mean > best
                else ( f'$\\mathbf{{ {mean:.3f} \\pm {std:.3f} }}$' )
                for (mean, std) in row_data
            ])
        elif highlight == 'max':
            row = ' & '.join([
                ( f'${mean:.3f} \\pm {std:.3f}$' ) if mean < second
                else ( f'$\\mathit{{ {mean:.3f} \\pm {std:.3f} }}$' ) if mean < best
                else ( f'$\\mathbf{{ {mean:.3f} \\pm {std:.3f} }}$' )
                for (mean, std) in row_data
            ])
        return row
    
    if not single_row:
        content = backreturn.join([
            f'{row_name} & ' + row_content(
                results[row_name], best[row_name][0], second[row_name][0]
            ) for row_name in row_names
        ])
    else:
        content = row_content(results, best[0], second[0])
        
    return f'''
        \\begin{{table}}[ht]
            \\caption{{
                {caption}
            }}
            \\centering
            \\begin{{tabular}}{{@{{}}{columns_preamble}@{{}}}}
                \\toprule
                {columns} \\\\
                \\midrule
                {content}\\\\
                \\bottomrule
            \\end{{tabular}}
            \\label{{
                table:{label}
            }}
        \\end{{table}}
    '''.strip()


def bootstrap(
        data: Dict[str, NDArray] | Dict[str, Dict[str, NDArray]],
        n_samples: int=1000
    ) -> Dict:
    def bootstrap_single_row(
            data: Dict[str, NDArray], n_samples: int=n_samples
        ):
        if len(list(data.values())[0].shape) == 1:
            data = {
                key: value.copy().reshape(1, -1)
                for key, value in data.items()
            }

        def bootstrap_sample(data, n_bootstrap: Optional[int]=None):
            if len(data.shape) == 1:
                data = data.copy().reshape(1, -1)
            if n_bootstrap is None:
                n_bootstrap = data.shape[-1]
            N, M = data.shape
            idx = np.random.randint(0, M, (N, n_bootstrap))
            sample = np.take_along_axis(data, idx, axis=1)
            return sample
        

        bootstrapped_data = {
            model: np.zeros((data[model].shape[0], n_samples)) for model in data
        }
        for model in data:
            for i in range(n_samples):
                bootstrapped_data[model][:, i] = np.mean(
                    bootstrap_sample(data[model]),
                    axis = 1
                )
        return bootstrapped_data
    
    # check if data keys are subset of TEX_MAPPER keys
    # i.e., check if data keys only correspond to methods
    # if yes, then bootstrap, else access method sub-dict.
    single_row = set(data) <= set(TEX_MAPPER)
    if single_row:
        return bootstrap_single_row(data)
    else:
        return {
            key: bootstrap_single_row(data[key]) for key in data
        }


def json_default(obj: Any):
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    raise TypeError(f'Unknown type: {type(obj)}.')


def save(
        obj: Any,
        fname: str,
        experiment: Experiment,
        format: Plot | Literal['pkl', 'json', 'tex'],
        **kwargs
    ):
    path = f'{ARTIFACTS_DIRECTORY}/{experiment}'
    
    if not os.path.exists(path):
        os.makedirs(path)
    
    try:
        if format == 'pkl':
            with open(f'{path}/{fname}.pkl', 'wb+') as file:
                pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)
        elif format == 'json':
            with open(f'{path}/{fname}.json', 'w+') as file:

                try:
                    json.dump(
                        obj,
                        file,
                        separators=(',', ':'),
                        sort_keys=True,
                        indent=4,
                        default=json_default
                    )
                except Exception as e:
                    logger.error(
                        f'Could not convert {fname} obj from exp {experiment} to json.'
                    )
                    raise e
                
        elif format == 'tex':
            with open(f'{path}/{fname}.tex', 'w+') as file:
                file.write(obj)
        elif format in typing.get_args(Plot):
            obj.savefig(
                f'{path}/{fname}.{format}',
                format=format,
                **kwargs
            )
        else:
            raise NotImplementedError(f'Save not implemented for {format} file.')
    except Exception as e:
        logger.error(f'Could not save file {fname}.{format} at path {path}.')
        raise e
    
    logger.info(f'Saved file {fname}.{format} at path {path}.')


def load(path: str):
    if not os.path.exists(path):
        raise ValueError(f'Path {path} does not exist.')
    
    format = path.split('.')[-1]
    assert format == 'pkl' or format == 'json', \
        f'Incorrect format {format} of file, can only accept pkl or json.'
    
    try:
        if format == 'pkl':
            with open(path, 'rb') as file:
                data = pickle.load(file)
        elif format == 'json':
            with open(path, 'r') as file:
                data = json.load(file)
        else:
            raise NotImplementedError(f'Load not implemented for {format} file.')
    except Exception as e:
        logger.error(f'Could not load data from file {path}.')
        raise e
    
    logger.info(f'Loaded data from file {path}.')
    return data


def fit_model(
        model, name, X, y, G, GX, hyperparameters=None, pbar_manager=None, da=None
    ):
    if not pbar_manager:
        return fit_model_nopbar(model, name, X, y, G, GX, hyperparameters, da)

    sgd_params = getattr(hyperparameters, 'sgd', dict())
    if name == 'ERM':
        model.fit(
            X=X, y=y, pbar_manager=pbar_manager, **sgd_params
        )
    elif name == 'DA+ERM':
        model.fit(
            X=GX, y=y, pbar_manager=pbar_manager, **sgd_params
        )
    elif 'DA+IVL' in name:
        if 'LCV' in name:
            G = discretize(G)
        model.fit(
            X=X, y=y, G=G, GX=GX, pbar_manager=None, **sgd_params
        )
    elif name in ('DA+IV', 'DRO', 'ICP', 'IRM', 'V-REx', 'MM-REx'):
        G = discretize(G)
        model.fit(
            X=GX, y=y, Z=G, pbar_manager=pbar_manager, **sgd_params
        )
    elif 'RICE' in name:
        X_rice, _, y_rice, _ = train_test_split(
            X, y, train_size=1.0/RICE_AUGMENTATIONS
        )
        model.fit(
            X=X_rice, y=y_rice,
            da=da, num_augmentations=RICE_AUGMENTATIONS,
            pbar_manager=pbar_manager, **sgd_params
        )
    else:
        raise ValueError(f'Model {name} not implemented.')


def fit_model_nopbar(model, name, X, y, G, GX, hyperparameters=None, da=None):
    sgd_params = getattr(hyperparameters, 'sgd', dict())
    if name in ('ERM', 'L1Janzing', 'L2Janzing'):
        model.fit(
            X=X, y=y, **sgd_params
        )
    elif name == 'DA+ERM':
        model.fit(
            X=GX, y=y, **sgd_params
        )
    elif 'DA+IVL' in name:
        if 'LCV' in name:
            G = discretize(G)
        model.fit(
            X=X, y=y, G=G, GX=GX, **sgd_params
        )
    elif name in ('DA+IV', 'DRO', 'ICP', 'IRM', 'V-REx', 'MM-REx'):
        G = discretize(G)
        model.fit(
            X=GX, y=y, Z=G, **sgd_params
        )
    elif 'RICE' in name:
        X_rice, _, y_rice, _ = train_test_split(
            X, y, train_size=1.0/RICE_AUGMENTATIONS
        )
        model.fit(
            X=X_rice, y=y_rice,
            da=da, num_augmentations=RICE_AUGMENTATIONS,
            **sgd_params
        )
    elif 'Kania&Wit' in name:
        model.fit(
            X=X, y=y, X_A=GX, y_A=y, **sgd_params
        )
    else:
        raise ValueError(f'Model {name} not implemented.')
