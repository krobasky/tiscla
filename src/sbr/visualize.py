# xxx make this a function
# plot and save the loss curve
import os
import matplotlib.pyplot as plt

def plot_loss_curve(history, 
                    figsize=(5,5),
                    metrics = ['loss','accuracy','val_accuracy'],
                    write_directory="data/images",
                    file_name="nn-loss-curve.png",
                    show_plot=True):
    """
    Plot a loss, accuracy curve. Assumes loss and accuracy were compiled into the model metrics.
    If this is running in a notebook, the `plt.show()` command doesn't matter and the plot will just who no matter what

    Args:
      history: history object returned from model.fit (or sbr.fit.multicategorical_model)
      figsize [(5,5)]: tuple for the size of the figure
      metrics []: traces to plot
      write_directory [['loss','accuracy']]: where to write out the figure (if None, nothing is saved)
      file_name ["data/images"]: override filename of figure to be written
      show [True]: if True, show to figure to display

    Returns:
      plots to display if show= True, saves image if write_directory not None

    Example usage:
      plot_loss_curve(figsize=(5,5)
                history, ['loss','accuracy','val_accuracy'],
                write_directory="data/images",
                show=True)

    """
    plt.figure(figsize=figsize)
    plt.title('Training metrics')
    plt.plot(history.history['loss'], label='loss')
    plt.ylabel('loss')
    for metric in metrics:
        if metric == "loss":
            continue
        plt.plot(history.history[metric], label=metric)
    plt.xlabel('Epochs')
    plt.legend()
    if write_directory is not None:
        os.makedirs(write_directory, exist_ok=True)
        plt.savefig(os.path.join(write_directory, file_name))
    if show_plot:
        plt.show()

