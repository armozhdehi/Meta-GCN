from sklearn.metrics import f1_score, roc_auc_score
import torch
import torch.nn.functional as F

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
def print_class_acc(output, labels, pre='valid'):
    pre_num = 0
    #print class-wise performance
    '''
    for i in range(labels.max()+1):
        
        cur_tpr = accuracy(output[pre_num:pre_num+class_num_list[i]], labels[pre_num:pre_num+class_num_list[i]])
        print(str(pre)+" class {:d} True Positive Rate: {:.3f}".format(i,cur_tpr.item()))

        index_negative = labels != i
        labels_negative = labels.new(labels.shape).fill_(i)
        
        cur_fpr = accuracy(output[index_negative,:], labels_negative[index_negative])
        print(str(pre)+" class {:d} False Positive Rate: {:.3f}".format(i,cur_fpr.item()))

        pre_num = pre_num + class_num_list[i]
    '''
    #ipdb.set_trace()
    if labels.max() > 1:
        auc_score = roc_auc_score(labels.detach(), F.softmax(output, dim=-1).detach(), average='macro', multi_class='ovr')
    else:
        auc_score = roc_auc_score(labels.detach(), F.softmax(output, dim=-1)[:,1].detach(), average='macro')
    macro_F = f1_score(labels.detach(), torch.argmax(output, dim=-1).detach(), average='macro')
    print(str(pre)+' current auc-roc score: {:f}, current macro_F score: {:f}'.format(auc_score,macro_F))
    return