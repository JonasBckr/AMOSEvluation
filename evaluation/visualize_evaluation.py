import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# create boxplots of all calsses

def load_data(score_path): 
    labels = {"0": "background", "1": "spleen", "2": "right kidney", "3": "left kidney", "4": "gall bladder", "5": "esophagus", "6": "liver", "7": "stomach", "8": "arota", "9": "postcava", "10": "pancreas", "11": "right adrenal gland", "12": "left adrenal gland", "13": "duodenum", "14": "bladder", "15": "prostate/uterus"}
    scores = pd.read_csv(scores_path)
    scores = scores.dropna()
    scores = scores.drop('Unnamed: 0', axis=1)
    scores = scores.drop('Unnamed: 0.1', axis=1)
    scores = scores.drop('sample', axis=1)
    scores = scores.drop('mean_precision', axis=1)
    scores = scores.drop('mean_iou', axis=1)
    
    for i in range(16):
        scores = scores.rename(columns={'dice' + str(i):'dice_' + labels[str(i)]})
        scores = scores.rename(columns={'accuracy' + str(i):'accuracy_' + labels[str(i)]})
        scores = scores.rename(columns={'sensitivity' + str(i):'sensitivity_' + labels[str(i)]})
        scores = scores.rename(columns={'specificity' + str(i):'specificity_' + labels[str(i)]})
    data = pd.melt(scores)

    data['value'] = data['value'].astype('float')
    return data


# plot figure for dice scores

def plot_dice(data, output_path):
    data_dice = data[data['variable'].str.contains('dice', na=False)]
    data_dice = data_dice.drop(data_dice[data_dice['variable'] == 'mean_dice'].index)
    data_dice['variable'] = data_dice['variable'].str.replace('dice_', '')
    data_dice = data_dice.rename(columns={'variable':'dice'})
    plt.figure(figsize=(15,5))
    plt.xticks(rotation=60, ha='right')
    plt.subplots_adjust(bottom=0.3)
    plt.rcParams['figure.dpi'] = 600

    boxplot = sns.boxplot(x='dice', y="value", data=data_dice)
    lines = boxplot.get_lines()
    categories = boxplot.get_xticks()

    for cat in categories:
        y = round(lines[4+cat*6].get_ydata()[0],2) 
        boxplot.text(
            cat, 
            y, 
            f'{y}', 
            ha='center', 
            va='center', 
            fontweight='semibold', 
            size=5,
            color='white',
            bbox=dict(facecolor='#828282', edgecolor='#828282')
        )
    boxplot.get_figure().savefig(output_path)

def plot_acc(data, output_path):
    data_accuracy = data[data['variable'].str.contains('accuracy', na=False)]
    data_accuracy = data_accuracy.drop(data_accuracy[data_accuracy['variable'] == 'mean_accuracy'].index)
    data_accuracy['variable'] = data_accuracy['variable'].str.replace('accuracy_', '')
    data_accuracy = data_accuracy.rename(columns={'variable':'accuracy'})
    plt.figure(figsize=(15,5))
    plt.xticks(rotation=60, ha='right')
    plt.subplots_adjust(bottom=0.3)
    plt.rcParams['figure.dpi'] = 600

    boxplot = sns.boxplot(x='accuracy', y="value", data=data_accuracy)
    lines = boxplot.get_lines()
    categories = boxplot.get_xticks()

    for cat in categories:
        y = round(lines[4+cat*6].get_ydata()[0],2) 
        boxplot.text(
            cat, 
            y, 
            f'{y}', 
            ha='center', 
            va='center', 
            fontweight='semibold', 
            size=5,
            color='white',
            bbox=dict(facecolor='#828282', edgecolor='#828282')
        )

    boxplot.get_figure().savefig(output_path)

def plot_sens(data, output_path):
    data_sensitivity = data[data['variable'].str.contains('sensitivity', na=False)]

    data_sensitivity = data_sensitivity.drop(data_sensitivity[data_sensitivity['variable'] == 'mean_sensitivity'].index)
    data_sensitivity['variable'] = data_sensitivity['variable'].str.replace('sensitivity_', '')
    data_sensitivity = data_sensitivity.rename(columns={'variable':'sensitivity'})
    plt.figure(figsize=(15, 5))
    plt.xticks(rotation=60, ha='right')
    plt.subplots_adjust(bottom=0.3)
    plt.rcParams['figure.dpi'] = 600
    
    boxplot = sns.boxplot(x='sensitivity', y="value", data=data_sensitivity)
    lines = boxplot.get_lines()
    categories = boxplot.get_xticks()

    for cat in categories:
        y = round(lines[4+cat*6].get_ydata()[0],2) 
        boxplot.text(
            cat, 
            y, 
            f'{y}', 
            ha='center', 
            va='center', 
            fontweight='semibold', 
            size=5,
            color='white',
            bbox=dict(facecolor='#828282', edgecolor='#828282')
        )
    boxplot.get_figure().savefig(output_path)

def plot_spec(data, output_path):
    data_specificity = data[data['variable'].str.contains('specificity', na=False)]
    data_specificity = data_specificity.drop(data_specificity[data_specificity['variable'] == 'mean_specificity'].index)
    data_specificity['variable'] = data_specificity['variable'].str.replace('specificity_', '')
    data_specificity = data_specificity.rename(columns={'variable':'specificity'})
    plt.figure(figsize=(15,5))
    plt.xticks(rotation=60, ha='right')
    plt.subplots_adjust(bottom=0.3)
    plt.rcParams['figure.dpi'] = 600

    boxplot = sns.boxplot(x='specificity', y="value", data=data_specificity)
    lines = boxplot.get_lines()
    categories = boxplot.get_xticks()

    for cat in categories:
        y = round(lines[4+cat*6].get_ydata()[0],2) 
        boxplot.text(
            cat, 
            y, 
            f'{y}', 
            ha='center', 
            va='center', 
            fontweight='semibold', 
            size=5,
            color='white',
            bbox=dict(facecolor='#828282', edgecolor='#828282')
        )
    boxplot.get_figure().savefig(output_path)


def plot_means(data, output_path):
    data_means = data[data['variable'].str.contains('mean', na=False)]
    data_means['variable'] = data_means['variable'].str.replace('mean_', '')
    data_means = data_means.rename(columns={'variable':'sample means across all classes'})
    plt.figure(figsize=(5,5))
    plt.rcParams['figure.dpi'] = 600

    boxplot = sns.boxplot(x='sample means across all classes', y="value", data=data_means)
    lines = boxplot.get_lines()
    categories = boxplot.get_xticks()

    for cat in categories:
        y = round(lines[4+cat*6].get_ydata()[0],2) 
        boxplot.text(
            cat, 
            y, 
            f'{y}', 
            ha='center', 
            va='center', 
            fontweight='semibold', 
            size=5,
            color='white',
            bbox=dict(facecolor='#828282', edgecolor='#828282')
        )
    boxplot.get_figure().savefig(output_path)


# miscnn
scores_path = '/data/miscnn_models/model0407/scores.csv'
data = load_data(scores_path)
output_path = "/data/scripts/images/boxplot_dice_3Dmiscnn.png"
plot_dice(data, output_path)
output_path = "/data/scripts/images/boxplot_acc_3Dmiscnn.png"
plot_acc(data, output_path)
output_path = "/data/scripts/images/boxplot_sensitivity_3Dmiscnn.png"
plot_sens(data, output_path)
output_path = "/data/scripts/images/boxplot_specificity_3Dmiscnn.png"
plot_spec(data, output_path)
output_path = "/data/scripts/images/boxplot_means_3Dmiscnn.png"
plot_means(data, output_path)

# nnunet
scores_path = '/data/scripts/nnupredictions/scores.csv'
data = load_data(scores_path)
output_path = "/data/scripts/images/boxplot_dice_nnunet.png"
plot_dice(data, output_path)
output_path = "/data/scripts/images/boxplot_acc_nnunet.png"
plot_acc(data, output_path)
output_path = "/data/scripts/images/boxplot_sensitivity_nnunet.png"
plot_sens(data, output_path)
output_path = "/data/scripts/images/boxplot_specificity_nnunet.png"
plot_spec(data, output_path)
output_path = "/data/scripts/images/boxplot_means_nnunet.png"
plot_means(data, output_path)