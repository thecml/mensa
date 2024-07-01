import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class _TFColor(object):
    """Enum of colors used in TF docs."""
    red = '#F15854'
    blue = '#5DA5DA'
    orange = '#FAA43A'
    green = '#60BD68'
    pink = '#F17CB0'
    brown = '#B2912F'
    purple = '#B276B2'
    yellow = '#DECF3F'
    gray = '#4D4D4D'
    def __getitem__(self, i):
        return [
            self.red,
            self.orange,
            self.green,
            self.blue,
            self.pink,
            self.brown,
            self.purple,
            self.yellow,
            self.gray,
        ][i % 9]
TFColor = _TFColor()

def make_bar_plot(combined_data, ybounds, methods, axis_labs, \
                   xtick_labs, colors, show_legend=False):
    num_methods = len(combined_data)
    _, ax1 = plt.subplots()
    
    for i in range(num_methods):
        for j in range(combined_data[i].shape[1]):
            data = combined_data[i]
            
            error_bars = data[[1, 2], :]
            error_bars[0, :] = data[0, :] - error_bars[0, :]
            error_bars[1, :] = error_bars[1, :] - data[0, :]
            
            xpos = (np.arange(0, (num_methods + 1) * data.shape[1], num_methods + 1) + i) 
            if j == 0:
                ax1.bar(xpos[j], data[0, j], yerr=error_bars[:, j].reshape(-1, 1), capsize=2, align='center', alpha=0.5, label=methods[i], color=colors[i])
            else:
                ax1.bar(xpos[j], data[0, j], yerr=error_bars[:, j].reshape(-1, 1), capsize=2, align='center', alpha=0.5, color=colors[i])

    plt.xticks([], [])
    if xtick_labs is None:
        xtick_scale = np.arange(combined_data[0].shape[1])    
        plt.xticks(xtick_scale * (num_methods + 1) + ((num_methods - 1) / 2), xtick_scale + 1)    
    else:
        plt.xticks(xtick_labs[0], xtick_labs[1]) 
        
    #ax1.legend(bbox_to_anchor=(0, 1.15), ncol=1, loc='upper left')  
    ax1.set_ylabel(axis_labs[0])
    ax1.yaxis.grid() 
    ax1.set_ylim(ybounds[0])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    if show_legend:
        plt.legend(loc="upper left")
    plt.show()