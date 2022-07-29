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
  """Plot a loss, accuracy curve. Assumes loss and accuracy were compiled into the model metrics.
  If this is running in a notebook, the `plt.show()` command doesn't matter and the plot will show no matter what

  Args:
    history: history object returned from model.fit (or sbr.fit.multicategorical_model)

    figsize: tuple for the size of the figure

    metrics: traces to plot

    write_directory: : where to write out the figure (if None, nothing is saved)

    file_name: override filename of figure to be written

    show: if True, show to figure to display

  Returns:
    plots to display if show= True, saves image if write_directory not None

  Example Usage:
    >>> plot_loss_curve(history=history, 
                figsize=(5,5),
                metrics = ['loss','accuracy','val_accuracy'],
                write_directory="data/images",
                file_name="nn-loss-curve.png",
                show_plot=True)
    >>> import matplotlib.pyplot as plt
    >>> import matplotlib.image as mpimg
    >>> img = mpimg.imread("nn-loss-curve.png")
    >>> plt.imshow(img);

    .. image:: images/loss_curve.png

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


'''
# Note: The following confusion matrix code is a remix of Scikit-Learn's 
# plot_confusion_matrix function - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_confusion_matrix.html
# and Made with ML's introductory notebook - https://github.com/GokuMohandas/MadeWithML/blob/main/notebooks/08_Neural_Networks.ipynb
''' 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import numpy as np
def plot_cm(y_test, y_pred, figsize=(10,10), labelsize=20, textsize=15, classes=None):
  """Makes a labelled confusion matrix comparing predictions and ground truth labels.

  If classes is passed, confusion matrix will be labelled, if not, integer class values
  will be used.

  Args:
    y_test: Array of truth labels (must be same shape as y_pred).

    y_pred: Array of predicted labels (must be same shape as y_true).

    figsize: Size of output figure (default=(10, 10)).

    label_size: Size of label text (default=20).

    text_size: Size of output figure text (default=15).

    classes: Array of class labels (e.g. string form). If `None`, integer labels are used.
  
  Returns:
    A labelled confusion matrix plot comparing y_test and y_pred.


  Example usage:
    >>> make_confusion_matrix(y_test=test_labels, # ground truth test labels
                              y_pred=y_preds, # predicted labels
                              classes=class_names, # array of class label names
                              figsize=(15, 15),
                              text_size=10)
   .. image:: images/example_cm.png


  """  
  import itertools

  # Create the confusion matrix
  cm = confusion_matrix(y_test, tf.round(y_pred))
  cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] # normalize it
  n_classes = cm.shape[0]

  # Let's prettify it
  fig, ax = plt.subplots(figsize=figsize)
  # Create a matrix plot
  cax = ax.matshow(cm, cmap=plt.cm.Blues) # https://matplotlib.org/3.2.0/api/_as_gen/matplotlib.axes.Axes.matshow.html
  fig.colorbar(cax)

  # Create classes

  if classes:
    labels = classes
  else:
    labels = np.arange(cm.shape[0])

  # Label the axes
  ax.set(title="Confusion Matrix",
       xlabel="Predicted label",
       ylabel="True label",
       xticks=np.arange(n_classes),
       yticks=np.arange(n_classes),
       #xticklabels=labels,
       yticklabels=labels)
  ax.xaxis.set_ticklabels(labels, rotation=45)

  # Set x-axis labels to bottom
  ax.xaxis.set_label_position("bottom")
  ax.xaxis.tick_bottom()

  # Adjust label size
  ax.xaxis.label.set_size(labelsize)
  ax.yaxis.label.set_size(labelsize)
  ax.title.set_size(labelsize)

  # Set threshold for different colors
  threshold = (cm.max() + cm.min()) / 2.

  # Plot the text on each cell
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#    plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
    plt.text(j, i, f"{cm_norm[i, j]:.2f}",
            horizontalalignment="center",
           color="white" if cm[i, j] > threshold else "black",
           size=textsize)
