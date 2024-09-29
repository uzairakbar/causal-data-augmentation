import os
import json
import torch
import random
import pickle
import typing
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import KBinsDiscretizer
from typing import Any, Literal, List, Dict, Optional

from src.sem.simulation.linear import COVARIATE_DIMENSION


Experiment = Literal[
    'linear_simulation',
    'nonlinear_simulation',
    'optical_device',
    'colored_mnist',
    'rotated_mnist'
]
Plot = Literal['png', 'pdf', 'ps', 'eps', 'svg']

PLOT_DPI: int=1200
PLOT_FORMAT: Plot='pdf'
ARTIFACTS_DIRECTORY: str='artifacts'
TEX_MAPPER: Dict[str, str] = {
    'Data': r'Data',
    'ERM': r'ERM',
    'DA+ERM': r'DA+ERM',
    'DA+UIV-a': r'DA+UIV--$\alpha$',
    'DA+UIV-5fold': r'DA+UIV--$\alpha^{\mathrm{5-fold}}$',
    'DA+UIV-LOLO': r'DA+UIV--$\alpha^{\mathrm{LOLO}}$',
    'DA+UIV-CC': r'DA+UIV--$\alpha^{\mathrm{CC}}$',
    'DA+UIV-Pi': r'DA+UIV--$\Pi$',
    'DA+UIV': r'DA+UIV',
    'DA+IV': r'DA+IV',
    'IRM': r'IRM',
    'AR': r'AR',
    'ICP': r'ICP',
    'DRO': r'DRO',
    'RICE': r'RICE',
    'V-REx': r'V-REx',
    'MM-REx': r'MM-REx',
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
        format: Plot=PLOT_FORMAT
    ):
    sns.set_style('darkgrid')
    colors = sns.color_palette()[:len(y)+1]
    fig = plt.figure()
    labels = []
    for i, (method, errors) in enumerate(y.items()):
        mean = errors.mean(axis = 1)

        label = TEX_MAPPER.get(method, method)
        if method in vertical_plots:
            label = f'average {label.split("--")[-1]}'
        
        labels.append(label)

        if method in vertical_plots:
            plt.axvline(x = mean.mean(), color=colors[i], label=label)
        else:
            plt.plot(x, mean, color=colors[i], label=label)
    
    if trivial_solution:
        label = f'$\\mathbf{{0}}_{{{COVARIATE_DIMENSION}}}$'
        labels.append(label)
        plt.axhline(y = 0.5**0.5, color = colors[-1], label=label)
        
    for i, (method, errors) in enumerate(y.items()):
        if method not in vertical_plots:
            low = np.percentile(errors, 2.5, axis=1)
            high = np.percentile(errors, 97.5, axis=1)
            plt.fill_between(x, low, high, color=colors[i], alpha = 0.1)
    
    plt.xlabel(xlabel), plt.ylabel(ylabel)
    plt.xlim([min(x), max(x)])
    plt.xscale(xscale)
    plt.legend(labels)
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


def box_plot(
        data: Dict[str, NDArray],
        fname: str,
        experiment: Experiment,
        xlabel: Optional[str]='Relative Error',
        ylabel: Optional[str]='Method',
        zlabel: Optional[str]='Augmentation',
        orient: Literal['h', 'v']='h',
        savefig: Optional[bool]=True,
        format: Plot=PLOT_FORMAT,
    ):

    def prepare_data_for_plotting(
            data: Dict[str, Dict[str, NDArray]]
        ) -> pd.DataFrame:
        records = []
        for augmentation, methods in data.items():
            for method, values in methods.items():
                for value in values.flatten():
                    records.append({
                        zlabel: augmentation,
                        ylabel: TEX_MAPPER.get(method, method),
                        xlabel: value
                    })
        df = pd.DataFrame.from_records(records)
        return df
    
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
    df = prepare_data_for_plotting(data)
    
    sns.set_style('darkgrid')
    fig = plt.figure()

    if orient == 'v':
        xlabel, ylabel = ylabel, xlabel
        plt.xticks(rotation=45)

    ax = sns.boxplot(
        y=ylabel, x=xlabel, hue=zlabel,
        data=df,
        orient=orient,
        showmeans=False,
        flierprops={"marker": "d"},
        showcaps=False,
    )
    
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


def grid_plot(
        data: Dict,
        fname: str,
        experiment: Experiment,
        savefig: Optional[bool]=True,
        format: Plot=PLOT_FORMAT
    ):
    functions = data.keys()
    methods = ['Data'] + ([
        method for method in data['abs'].keys() if 'ERM' in method or 'DA' in method or 'IV' in method
    ])
    
    sns.set_style('darkgrid')
    colors = sns.color_palette()[:3]
    fig, axs = plt.subplots(
        len(functions), len(methods), figsize=(3*len(methods), 3*len(functions)), sharex=True, sharey=False
    )
    for i, function in enumerate(functions):
        for j, method in enumerate(methods):
            x = data[function]['x']
            y = data[function]['y']
            
            if method == 'Data':
                axs[i, j].scatter(
                    data[function]['x_data'],
                    data[function]['y_data'],
                    color=colors[-1],
                    alpha=0.4
                )
            else:
                mean = data[function][method].mean(axis=1)
                low = np.percentile(data[function][method], 2.5, axis=1)
                high = np.percentile(data[function][method], 97.5, axis=1)

                axs[i, j].plot(x, mean, color=colors[0])
                axs[i, j].fill_between(x, low, high, color=colors[0], alpha=0.1)
            
            label = TEX_MAPPER.get(method, method)

            axs[i, j].plot(x, y, color=colors[1])
            if not i:
                axs[i, j].set_title(label)
            if not j:
                axs[i, j].set_ylabel(function)
            if j:
                axs[i, j].yaxis.set_ticklabels([])

            y_range = max(y) - min(y)
            y_pad = (y_range/2)*1.5
            axs[i, j].set_xlim([min(x) - 0.333, max(x) + 0.333])
            axs[i, j].set_ylim([min(y) - y_pad, max(y) + y_pad])
    plt.tight_layout(pad = 0.333)
    plt.show()
    if savefig:
        save(
            obj=fig,
            fname=fname,
            experiment=experiment,
            format=format,
            dpi=PLOT_DPI,
            bbox_inches='tight'
        )


def tex_table(
        data: Dict,
        label: str,
        caption: str,
        highlight: Literal['min', 'max']='min',
        decimals: int=3
    ):
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
            columns = {col: data[row][col] for col in TEX_MAPPER.keys() if col in data[row]}
            results[row] = [np.round((np.mean(v), np.std(v)), decimals) for v in columns.values()]
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


def fit_model(model, name, X, y, G, GX, hyperparameters=None, pbar_manager=None, da=None):
    if not pbar_manager:
        return fit_model_nopbar(model, name, X, y, G, GX, hyperparameters, da)

    erm_params = getattr(hyperparameters, 'erm', dict())
    gmm_params = getattr(hyperparameters, 'gmm', dict())
    if name == 'ERM':
        model.fit(
            X=X, y=y, pbar_manager=pbar_manager, **erm_params
        )
    elif name == 'DA+ERM':
        model.fit(
            X=GX, y=y, pbar_manager=pbar_manager, **erm_params
        )
    elif 'DA+UIV' in name:
        if 'LOLO' in name:
            G = discretize(G)
        model.fit(
            X=X, y=y, G=G, GX=GX, pbar_manager=None, **gmm_params
        )
    elif 'DA+IV' == name:
        model.fit(
            X=GX, y=y, Z=G, pbar_manager=pbar_manager, **gmm_params
        )
    elif 'AR' in name:
        model.fit(
            X=GX, y=y, Z=G, pbar_manager=pbar_manager, **erm_params
        )
    elif name in ('DRO', 'ICP', 'IRM', 'V-REx', 'MM-REx'):
        G = discretize(G)
        model.fit(
            X=GX, y=y, Z=G, pbar_manager=pbar_manager, **erm_params
        )
    elif 'RICE' in name:
        model.fit(
            X=X, y=y, da=da, pbar_manager=pbar_manager, **erm_params
        )
    else:
        raise ValueError(f'Model {name} not implemented.')


def fit_model_nopbar(model, name, X, y, G, GX, hyperparameters=None, da=None):
    erm_params = getattr(hyperparameters, 'erm', dict())
    gmm_params = getattr(hyperparameters, 'gmm', dict())
    if name == 'ERM':
        model.fit(
            X=X, y=y, **erm_params
        )
    elif name == 'DA+ERM':
        model.fit(
            X=GX, y=y, **erm_params
        )
    elif 'DA+UIV' in name:
        if 'LOLO' in name:
            G = discretize(G)
        model.fit(
            X=X, y=y, G=G, GX=GX, **gmm_params
        )
    elif 'DA+IV' == name:
        model.fit(
            X=GX, y=y, Z=G, **gmm_params
        )
    elif 'AR' in name:
        model.fit(
            X=GX, y=y, Z=G, **erm_params
        )
    elif name in ('DRO', 'ICP', 'IRM', 'V-REx', 'MM-REx'):
        G = discretize(G)
        model.fit(
            X=GX, y=y, Z=G, **erm_params
        )
    elif 'RICE' in name:
        model.fit(
            X=X, y=y, da=da, **erm_params
        )
    else:
        raise ValueError(f'Model {name} not implemented.')
