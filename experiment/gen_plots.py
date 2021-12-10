import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import numpy as np

def get_colors():
    # return np.array([
    #     [0.568, 0.721, 0.741],  # economist mid-green
    #     [0.831, 0.866, 0.866],  # economist light-gray
    #     [0.290, 0.290, 0.290],  # economist dark gray
    #     # [0.890, 0.070, 0.043],  # economist red
    #     [0.800, 0.070, 0.043],  # economist red
    #     [0.541, 0.733, 0.815],  # economist mid-blue
    #     [0.985, 0.726, 0],      # yellow
    #     [100/255, 100/255, 100/255],# economist dark gray
    #     [0, 0, 0],              # black
    #     [1/255, 82/255, 109/255],   # economist dark blue
    #     # [41/255, 176/255, 14/255]   # economist green
    #     [70/255, 140/255, 140/255],   # economist green
    #     [0/255, 102/255, 255/255], # another blue
    #     [255/255, 102/255, 0/255], # orange
    # ])
    return np.array([
        [46/255, 125/255, 158/255],  # blue
        [241/255, 193/255, 109/255], # yellow
        [229/255, 120/255, 114/255], # red
        [128/255, 50/255, 131/255], # magenta
        [50/255, 50/255, 50/255], # dark gray
        [1, 1, 1] # white
    ])

def color_bars(axes, colors):
    # Iterate over each subplot
    for ax in axes:

        # Pull out the dark and light colors for
        # the current subplot
        myblue = colors[0]
        myyellow = colors[1]
        myred = colors[2]
        magenta = colors[3]
        dark_gray = colors[4]
        white = colors[5]

        # These are the patches (matplotlib's terminology
        # for the rectangles corresponding to the bars)
        p1, p2, p3 = ax.patches

        # The first bar gets the dark color
        p1.set_color(myyellow)
        p1.set_edgecolor(white)
        p1.set_hatch('///')
        
        # The second bar gets the light color, plus
        # hatch marks int he dark color
        p2.set_color(myred)
        p2.set_edgecolor(white)
        # p2.set_hatch('///')

        p3.set_color(myblue)
        p3.set_edgecolor(white)
        # p3.set_hatch('..')
        # p3.set_hatch('---')

        # p4.set_color(darkgray_color)
        # p4.set_edgecolor(line_color)
        # p4.set_hatch('----')

def get_color_palette(colors):

    dark_color = colors[0]
    light_color = colors[3]
    line_color = colors[6]
    blue_color = colors[4]
    yellow_color = colors[5]
    darkgray_color = colors[6]
    darkblue_color = colors[8]
    green_color = colors[9]
    
    palette=sns.color_palette([green_color, light_color, dark_color, darkgray_color])

    return palette

def set_plot_style():
    plt.style.use(['seaborn-white', 'seaborn-paper'])
    # plt.style.use(['fivethirtyeight'])
    matplotlib.rc('font', family='Times')
    matplotlib.rcParams['hatch.linewidth'] = 2.0 
    # sns.set_context('paper')
    sns.set(font='serif', font_scale=1.8)
    sns.set_style('white', {
        'font.family': 'serif',
        'font.serif': ['Times', 'Palatino', 'serif']
    })


def plot_exp_results(results, ncat=4, labels=['Basis', 'D1', 'D2']):
    ### Divide shears and rotations
    results_shears = results.drop(results[(results.condition == 'rot1') | (results.condition == 'rot2')].index)
    results_rots = results.drop(results[(results.condition == 'shear1') | (results.condition == 'shear2')].index)

    ### Plot Same and Diff at different stops
    plt.figure()
    set_plot_style()
    plt.rcParams['font.size'] = '11'
    plt.rcParams['figure.dpi'] = 200

    colors = get_colors()
    myblue = colors[0]
    myred = colors[2]
    ax = sns.pointplot(data=results_shears, x='condition', y='correct', order=['base', 'shear1', 'shear2'], color=myred, markers='o', linewidth=4, scale=2)
    ax = sns.pointplot(data=results_rots, x='condition', y='correct', order=['base', 'rot1', 'rot2'], color=myblue, linestyles='--', markers='^', linewidth=4, scale=2, ax=ax)
    # ax.set_xticklabels(labels, rotation=90, position=[0, 0, 0, 0])
    ax.set_xticklabels(labels, rotation=0)
    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles, ['Shear', 'Rotation'], fontsize=16, title_fontsize=20)
    ax.hlines((1/ncat), ax.get_xlim()[0], ax.get_xlim()[1], linestyle='--', linewidth=1, colors='r') # chance line
    ax.set_xlabel("")
    ax.set_ylabel('Accuracy')
    ax.set_ylim((0,1.05))
    sns.despine(ax=ax, left=False)
    custom_lines = [Line2D([0], [0], color=myred, lw=4),
                    Line2D([0], [0], color=myblue, lw=4, linestyle='--')]
    ax.legend(custom_lines, ['shear', 'rotation'], loc='lower left')

    ### set the figure title
    fig = plt.gcf()
    fig.set_size_inches(6.5, 7)
    fig.tight_layout()

    out_file = 'exp_results.png'
    fig.savefig(out_file)
    fig.savefig(out_file)

def plot_sim_results(results, test_type='no_rot', model_name='vgg', ncat=4, labels=['Basis', 'D1', 'D2']):
    ### Select rows from results that correspond to experimental conditions
    selected_shears = ['sh_1', 'sh_4', 'sh_8']
    selected_rots = ['dist_1', 'dist_4', 'dist_8']
    results = results[results['shear'].isin(selected_shears)]
    results = results[results['rotation'].isin(selected_rots)]
    results = results.drop(results[(results.shear == 'sh_4') & (results.rotation == 'dist_4')].index) # we don't test sh_4 && dist_4

    ### Divide shears and rotations
    ### When checking effect of shear don't consider rotation=4 and rotation=8 in rows where shear=sh_1
    ### When checking effect of rotation don't consider shear=4 and shear=8 in rows where rotation=dist_1
    results_shears = results.drop(results[(results.rotation == 'dist_4') | (results.rotation == 'dist_8')].index)
    results_rots = results.drop(results[(results.shear == 'sh_4') | (results.shear == 'sh_8')].index)

    ### Plot Same and Diff at different stops
    plt.figure()
    set_plot_style()
    plt.rcParams['font.size'] = '12'
    plt.rcParams['figure.dpi'] = 200

    colors = get_colors()
    myblue = colors[0]
    myred = colors[2]
    ax = sns.pointplot(data=results_shears, x='shear', y='correct', order=['sh_1', 'sh_4', 'sh_8'], color=myred, markers='o', linewidth=4, scale=2)
    ax = sns.pointplot(data=results_rots, x='rotation', y='correct', order=['dist_1', 'dist_4', 'dist_8'], color=myblue, linestyles='--', markers='^', scale=2, linewidth=4, ax=ax)
    # ax.set_xticklabels(labels, rotation=90, position=[0, 0, 0, 0])
    ax.set_xticklabels(labels, rotation=0)
    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles, ['Shear', 'Rotation'], fontsize=16, title_fontsize=20)
    ax.hlines((1/ncat), ax.get_xlim()[0], ax.get_xlim()[1], linestyle='--', linewidth=1, colors='k') # chance line
    ax.set_xlabel("")
    ax.set_ylabel('Accuracy', labelpad=7.0)
    ax.set_ylim((0,1.05))
    sns.despine(ax=ax, left=False)
    # plt.legend(labels=['shear', 'rotation'])
    custom_lines = [Line2D([0], [0], color=myred, lw=4),
                    Line2D([0], [0], color=myblue, lw=4, linestyle='--')]
    ax.legend(custom_lines, ['shear', 'rotation'], loc='lower left')

    ### set the figure title
    fig = plt.gcf()
    fig.set_size_inches(6.5, 7)
    fig.tight_layout()

    if test_type == 'no_rot':
        out_file = os.path.join('experiment', model_name + '_sim_results.png')
    elif test_type == 'some_rot':
        out_file = os.path.join('experiment', model_name + '_sim_results_train_rots.png')
    elif test_type == 'all_rot':
        out_file = os.path.join('experiment', model_name + '_sim_results_teach_rot.png')
    fig.savefig(out_file)