import os
import torch
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def relative_sq_error(W, What):
    sqnorm = lambda x: (x**2).sum()
    error = sqnorm(What - W)
    relative_error = error / (error + sqnorm(W))
    return relative_error


def set_seed(seed = 42):
    np.random.seed(seed)
    
    random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    print(f"Random seed set as {seed}")


def sweep_plot(x, y,
               xlabel,
               ylabel="RSE",
               xscale="linear",
               vertical_plots=[],
               save=True,
               trivial_solution=True):
    sns.set_style("darkgrid")
    colors = sns.color_palette()[:len(y)+1]
    fig = plt.figure()
    labels = []
    for i, (method, errors) in enumerate(y.items()):
        mean = errors.mean(axis = 1)

        if "DAIV" in method:
            alpha = "" if (method == "DAIV") else method.split("+")[-1]
            label_prefix = "average " if (method in vertical_plots) else "DAIV-"
            label = label_prefix + fr"$\alpha^{{\mathrm{{\mathsf{{{alpha}}}}}}}$"
        else:
            label = method
        labels.append(label)

        if method in vertical_plots:
            plt.axvline(x = mean.mean(), color=colors[i], label=label)
        else:
            plt.plot(x, mean, color=colors[i], label=label)
    
    if trivial_solution:
        label = r"$\mathbf{0}_{30}$"
        labels.append(label)
        plt.axhline(y = 0.5, color = colors[-1], label=label)
        
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
        fname = "".join(c for c in xlabel if c.isalnum()) + "_sweep"
        fig.savefig(f"assets/{fname}.pdf", format="pdf", dpi=1200)


def grid_plot(data, save=True, fname="nonlinear_simulation"):
    label = {
        "Data": "Data",
        "ERM": "ERM",
        "DA+ERM": "DA+ERM",
        "DA+IV": "DA+IV",
        "DAIV+LOO": r"DAIV-$\alpha^{\mathrm{\mathsf{LOO}}}$",
        "DAIV+LOLO": r"DAIV-$\alpha^{\mathrm{\mathsf{LOLO}}}$",
    }

    functions = data.keys()
    methods = ["Data"] + ([
        method for method in data["abs"].keys() if "ERM" in method or "DA" in method or "IV" in method
    ])
    
    sns.set_style("darkgrid")
    colors = sns.color_palette()[:3]
    fig, axs = plt.subplots(
        len(functions), len(methods), figsize=(18, 9), sharex=True, sharey=False
    )
    for i, function in enumerate(functions):
        for j, method in enumerate(methods):
            x = data[function]["x"]
            y = data[function]["y"]
            
            if method == "Data":
                axs[i, j].scatter(
                    data[function]["x_data"],
                    data[function]["y_data"],
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
        fig.savefig(
            f"assets/{fname}.pdf", format="pdf", dpi=1200, bbox_inches="tight"
        )


def tex_table(data,
              fname,
              title,
              decimals=3):
    label = {
        "ERM": "ERM",
        "DA+ERM": "DA+ERM",
        "DAIV+LOO": "DAIV--$\\alpha^{\\text{LOO}}$",
        "DAIV+LOLO": "DAIV--$\\alpha^{\\text{LOLO}}$",
        "DAIV+CC": "DAIV--$\\alpha^{\\text{CC}}$",
        "DA+IV": "DA+IV",
    }
    
    if "ERM" in data:
        row_names = None
        results = [np.round((np.mean(v), np.std(v)), decimals) for v in data.values()]
        minimum = min(results, key = lambda v : v[0])[0]
        column_names = [label[k] for k in data]
    else:
        row_names = list(data.keys())
        results = {}
        minimum = {}
        for row in row_names:
            columns = {col: data[row][col] for col in label.keys() if col in data[row]}
            results[row] = [np.round((np.mean(v), np.std(v)), decimals) for v in columns.values()]
            minimum[row] = min(results[row], key = lambda v : v[0])[0]
        column_names = [label[k] for k in data[row_names[0]]]
    
    with open(f"assets/{fname}.tex", "w+") as f:
        backreturn = "\\\\\n" + " "*8

        num_columns = len(column_names) + int(row_names is not None)
        columns_preamble = " ".join(["c"]*num_columns)

        columns = " & ".join(column_names)
        if row_names is not None:
            columns = " & " + columns
        
        def row_content(row_data, minimum):
            row = " & ".join([
                f"${mean:.3f}\\pm {std:.3f}$" if mean > minimum
                else ("$\\mathbf{ "+f"{mean:.3f}\\pm {std:.3f}"+" }$")
                for (mean, std) in row_data
            ])
            return row
        
        if row_names is not None:
            content = backreturn.join([
                f"{row_name} & " + row_content(results[row_name], minimum[row_name])
                for row_name in row_names
            ])
        else:
            content = row_content(results, minimum)
        
        f.write(f"""
                \\begin{{table}}[ht]
                    \\caption{{
                        {title}
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
                    """.strip())

