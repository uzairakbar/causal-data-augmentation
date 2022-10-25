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

