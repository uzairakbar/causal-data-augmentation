import os
import json
import torch
import random
import pickle
import numpy as np
import seaborn as sns
from loguru import logger
import matplotlib.pyplot as plt
from typing import Any, Literal, List, Dict, Optional


PLOT_DPI=1200
PLOT_FORMAT='pdf'
ARTIFACTS_DIRECTORY='artifacts'


def relative_sq_error(W, What) -> float:
    sqnorm = lambda x: (x**2).sum()
    error = sqnorm(What - W)
    relative_error = error / (error + sqnorm(W))
    return relative_error


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
        ylabel: str='$\sqrt{{\mathrm{{\mathsf{{RSE}}}}}}$',
        xscale: Literal['linear', 'log']='linear',
        vertical_plots: List=[],
        save: bool=True,
        trivial_solution: bool=True
    ):
    sns.set_style('darkgrid')
    colors = sns.color_palette()[:len(y)+1]
    fig = plt.figure()
    labels = []
    for i, (method, errors) in enumerate(y.items()):
        mean = errors.mean(axis = 1)

        if 'DAIVPi' == method:
            label = fr'DA+UIV-$\Pi$'
        elif 'DAIV' == method:
            label = fr'DA+UIV'
        elif 'DAIV+LOO' == method:
            if method in vertical_plots:
                label = 'average '+fr'-$\alpha^{{\mathrm{{\mathsf{{5-fold}}}}}}$'
            else:
                label = 'DA+UIV'+fr'-$\alpha^{{\mathrm{{\mathsf{{5-fold}}}}}}$'
        elif 'DAIV' in method:
            alpha = '' if (method == 'DAIValpha') else method.split('+')[-1]
            label_prefix = 'average ' if (method in vertical_plots) else 'DA+UIV'
            label = label_prefix + fr'-$\alpha^{{\mathrm{{\mathsf{{{alpha}}}}}}}$'
        else:
            label = method
        labels.append(label)

        if method in vertical_plots:
            plt.axvline(x = mean.mean(), color=colors[i], label=label)
        else:
            plt.plot(x, mean, color=colors[i], label=label)
    
    if trivial_solution:
        label = r'$\mathbf{0}_{30}$'
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
    if save:
        fname = ''.join(c for c in xlabel if c.isalnum()) + '_sweep'
        if not os.path.exists(ARTIFACTS_DIRECTORY):
            os.mkdir(ARTIFACTS_DIRECTORY)
        fig.savefig(
            f'{ARTIFACTS_DIRECTORY}/{fname}.{PLOT_FORMAT}',
            format=PLOT_FORMAT,
            dpi=PLOT_DPI
        )


def box_plot(
        data: Dict,
        xlabel: str='$\sqrt{{\mathrm{{\mathsf{{RSE}}}}}}$',
        ylabel: str='method',
        fname: str='optical_device',
        save: bool=True
    ):
    if len(list(data.values())[0].shape) > 1:
        data = {
            key: value.copy().flatten()
            for key, value in data.items()
        }

    sns.set_style('darkgrid')
    fig = plt.figure()
    ax = sns.boxplot(data=list(data.values()), orient='h', showmeans=True)
    labels = []
    for method in data.keys():
        if method in ['DAIVpi', 'mmDAIV', 'DAIVP__', 'DAIV', 'DAIVP_']:
            mapper = {
                'mmDAIV': 'mmDAIV',
                'pDAIV': 'pDAIV',
                'DAIVpi': 'DA+UIV-$\Pi$',
                'DAIV': 'DA+UIV',
                'DA+IV': 'DA+IV',
            }
            label = mapper[method]
        elif 'DAIV' in method:
            alpha = '' if (method == 'DAIV') else method.split('+')[-1]
            label = fr'DA+UIV-$\alpha^{{\mathrm{{\mathsf{{{alpha}}}}}}}$'
        else:
            label = method
        labels.append(label)
    ax.set(yticklabels=labels)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    plt.tight_layout()
    plt.show()
    if save:
        if not os.path.exists(ARTIFACTS_DIRECTORY):
            os.mkdir(ARTIFACTS_DIRECTORY)
        fig.savefig(
            f'{ARTIFACTS_DIRECTORY}/{fname}.{PLOT_FORMAT}',
            format=PLOT_FORMAT,
            dpi=PLOT_DPI
        )


def grid_plot(
        data: Dict,
        save: bool=True,
        fname: str='nonlinear_simulation'
    ):
    label = {
        'Data': 'Data',
        'ERM': 'ERM',
        'DA+ERM': 'DA+ERM',
        'DA+IV': 'DA+IV',
        'mmDAIV': 'mmDAIV',
        'DAIV+LOO': r'DAIV-$\alpha^{\mathrm{\mathsf{LOO}}}$',
        'DAIV+LOLO': r'DAIV-$\alpha^{\mathrm{\mathsf{LOLO}}}$',
    }

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
                
            axs[i, j].plot(x, y, color=colors[1])
            if not i:
                axs[i, j].set_title(label[method])
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
    if save:
        if not os.path.exists(ARTIFACTS_DIRECTORY):
            os.mkdir(ARTIFACTS_DIRECTORY)
        fig.savefig(
            f'{ARTIFACTS_DIRECTORY}/{fname}.{PLOT_FORMAT}',
            format=PLOT_FORMAT,
            dpi=PLOT_DPI,
            bbox_inches='tight'
        )


def tex_table(
        data: Dict,
        fname: str,
        caption: str,
        highlight: Literal['min', 'max']='min',
        decimals: int=3
    ):
    label = {
        'ERM': 'ERM',
        'DA+ERM': 'DA+ERM',
        'DAIV+LOO': 'DA+UIV--$\\alpha^{\\text{LOO}}$',
        'DAIV+LOLO': 'DA+UIV--$\\alpha^{\\text{LOLO}}$',
        'DAIV+CC': 'DA+UIV--$\\alpha^{\\text{CC}}$',
        'mmDAIV': 'mmDAIV',
        'pDAIV': 'pDAIV',
        'DAIVpi': 'DA+UIV--$\Pi$',
        'DAIV': 'DA+UIV',
        'DA+IV': 'DA+IV',
    }
    
    if 'ERM' in data:
        row_names = None
        results = [np.round((np.mean(v), np.std(v)), decimals) for v in data.values()]
        if highlight == 'min':
            best = min(results, key = lambda v : v[0])[0]
        elif highlight == 'max':
            best = max(results, key = lambda v : v[0])[0]
        column_names = [label[k] for k in data]
    else:
        row_names = list(data.keys())
        results = {}
        best = {}
        for row in row_names:
            columns = {col: data[row][col] for col in label.keys() if col in data[row]}
            results[row] = [np.round((np.mean(v), np.std(v)), decimals) for v in columns.values()]
            if highlight == 'min':
                best[row] = min(results[row], key = lambda v : v[0])[0]
            elif highlight == 'max':
                best[row] = max(results[row], key = lambda v : v[0])[0]
        column_names = [label[k] for k in data[row_names[0]]]
    
    with open(f'{ARTIFACTS_DIRECTORY}/{fname}.tex', 'w+') as f:
        backreturn = '\\\\\n' + ' '*8

        num_columns = len(column_names) + int(row_names is not None)
        columns_preamble = ' '.join(['c']*num_columns)

        columns = ' & '.join(column_names)
        if row_names is not None:
            columns = ' & ' + columns
        
        def row_content(row_data, best):
            if highlight == 'min':
                row = ' & '.join([
                    f'${mean:.3f}\\pm {std:.3f}$' if mean > best
                    else ('$\\mathbf{ '+f'{mean:.3f}\\pm {std:.3f}'+' }$')
                    for (mean, std) in row_data
                ])
            elif highlight == 'max':
                row = ' & '.join([
                    f'${mean:.3f}\\pm {std:.3f}$' if mean < best
                    else ('$\\mathbf{ '+f'{mean:.3f}\\pm {std:.3f}'+' }$')
                    for (mean, std) in row_data
                ])
            return row
        
        if row_names is not None:
            content = backreturn.join([
                f'{row_name} & ' + row_content(results[row_name], best[row_name])
                for row_name in row_names
            ])
        else:
            content = row_content(results, best)
        
        f.write(f'''
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
                    \\label{{table:nonlin}}
                \\end{{table}}
                    '''.strip())


def bootstrap(data: Dict, n_samples: int=1000) -> Dict:
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


def json_default(obj: Any):
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    raise TypeError(f'Unknown type: {type(obj)}.')


def save(obj: Dict[str, Any], fname: str, format: Literal['pkl', 'json']='pkl'):
    if format == 'pkl':
        with open(f'{ARTIFACTS_DIRECTORY}/{fname}.pkl', 'wb+') as file:
            pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)
    elif format == 'json':
        with open(f'{ARTIFACTS_DIRECTORY}/{fname}.json', 'w+') as file:
            json.dump(
                obj,
                file,
                separators=(',', ':'),
                sort_keys=True,
                indent=4,
                default=json_default
            )
