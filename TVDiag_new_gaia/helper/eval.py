import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score



def RCA_eval(root_logit, num_nodes_list, roots):
    print("1111111111111111111111111111111111")
    res = {
        "HR@1": [],
        "HR@2": [],
        "HR@3": [],
        "HR@4": [],
        "HR@5": [],
        "HR@6": [],
        "HR@7": [],
        "HR@8": [],
        "HR@9": [],
        "HR@10": [],
        "HR@11": [],
        "HR@12": [],
        "HR@13": [],
        "HR@14": [],
        "HR@15": [],
        "HR@16": [],
        "HR@17": [],
        "HR@18": [],
        "HR@19": [],
        "HR@20": [],
        "HR@21": [],
        "HR@22": [],
        "HR@23": [],
        "HR@24": [],
        "HR@25": [],
        "HR@26": [],
        "HR@27": [],
        "HR@28": [],
        "HR@29": [],
        "HR@30": [],
        "HR@31": [],
        "HR@32": [],
        "HR@33": [],
        "HR@34": [],
        "HR@35": [],
        "HR@36": [],
        "HR@37": [],
        "HR@38": [],
        "HR@39": [],
        "HR@40": [],
        "MRR@3": []
    }  
    
    start_idx = 0
    for idx, num_nodes in enumerate(num_nodes_list):
        end_idx = start_idx + num_nodes
        node_logits = root_logit[start_idx : end_idx].reshape(1, -1)
        root = roots[start_idx : end_idx].tolist().index(1)

        _, sorted_indices = torch.sort(node_logits, descending=True)
        for j in range(1, 40):
            # HR@k
            if root in sorted_indices.flatten()[:j]:
                res[f"HR@{j}"].append(1)
                print("yes")
            else:
                res[f"HR@{j}"].append(0)
                print("no")
        # MRR
        rank = (sorted_indices == root).nonzero(as_tuple=True)[1].item() + 1
        if rank <= 3:
            res["MRR@3"].append(1 / rank)
        else:
            res["MRR@3"].append(0)

        start_idx += num_nodes
    for k in range(1, 40):
        res[f'HR@{k}'] = np.sum(res[f'HR@{k}'])/len(num_nodes_list)
    res['MRR@3'] = np.sum(res['MRR@3'])/len(num_nodes_list)
    print(res)
    return res
    
        
def FTI_eval(output, target, k=5):
    res = {"pre": [], "rec": [], "f1": []}
    res['pre']=precision(output, target, k)
    res['rec']=recall(output, target, k)
    res['f1']=2 * res['pre'] * res['rec'] / (res['pre'] + res['rec'])
    return res



def target_rank(output, target, k=10):
    _, pred = output.topk(k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    ranks = []
    for col in range(correct.size(1)):
        try:
            idx=torch.where(correct[:, col] == target[col])[0].item() + 1
        except:
            idx=10
        ranks.append(idx)
    
    return ranks


def precision(output, target, k=5):
    _, pred = output.topk(k, 1, True, True)
    y_pred = pred.cpu().detach().numpy()
    y_true = target.cpu().detach().numpy().reshape(-1, 1)
    pre = precision_score(y_true, y_pred[:, 0], average='weighted')

    return pre


def recall(output, target, k=5):
    _, pred = output.topk(k, 1, True, True)
    y_pred = pred.cpu().detach().numpy()
    y_true = target.cpu().detach().numpy().reshape(-1, 1)
    rec = recall_score(y_true, y_pred[:, 0], average='weighted')

    return rec


def f1score(output, target, k=5):
    _, pred = output.topk(k, 1, True, True)
    y_pred = pred.cpu().detach().numpy()
    y_true = target.cpu().detach().numpy().reshape(-1, 1)
    f1 = f1_score(y_true, y_pred[:, 0], average='weighted')

    return f1