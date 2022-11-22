import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def prob_to_class(outputs): # outputs the output of the classifier to label
    res = []
    for row in outputs:
        res.append(row.argmax())
    return torch.tensor(np.array(res))

def confusion_matrix_vis(outputs, labels):
    flatten_labels = labels.reshape(-1)
    predicted_classes = prob_to_class(outputs)
    classes = range(outputs.shape[-1])
    cm = confusion_matrix(predicted_classes.numpy(), flatten_labels)
    # cm = confusion_matrix_vis(outputs[test_idx], labels[test_idx])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.array(classes))
    disp.plot()
    plt.show() 