# utils.py
import os
import numpy as np
import scipy
from sklearn.metrics import confusion_matrix

def get_save_dir(arg):
    save_path = "./outs/" + arg.save_dir
    if os.path.exists(save_path) is False:
        os.mkdir(save_path)
    return save_path

def RVD(output, target):
    # Relative Volume Difference
    output_sum = output.sum()
    target_sum = target.sum()
    if output_sum == target_sum:
        return 1
    
    score = (output_sum - target_sum) / target_sum
    # Higher is Better
    return -score

def get_roc_pr(tn, fp, fn, tp):
    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 1
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 1

    precision = tp / (tp + fp) if (tp + fp) != 0 else 1
    recall    = tp / (tp + fn) if (tp + fn) != 0 else 1

    # f1 = 2 * precision * recall / (precision + recall) # if (precision + recall) != 0 else 1
    f1      = (2 * tp) / ((2 * tp) + fp + fn) if ((2 * tp) + fp + fn) != 0 else 1 
    jaccard = tp       / (tp + fp + fn)       if (tp + fp + fn)       != 0 else 1

    return sensitivity, 1 - specificity, precision, recall, f1, jaccard
    

def slice_threshold(_np, th):
    return (_np >= th).astype(int)
    
def image_save(save_path, *args):
    total = np.concatenate(args, axis=1)    
    np.save(save_path +'.npy', total)
    scipy.misc.imsave(save_path + '.jpg', total)
    """
    post_fix = ["input", "target", "output"]
    for i, post_fix in enumerate(post_fix):
        npy = args[i]
        np.save(save_path + "_%s.npy"%(post_fix), npy)
        scipy.misc.imsave(save_path + '_%s.jpg'%(post_fix), npy)
    """

def slack_alarm(send_id, send_msg="Train Done"):
    """
    send_id : slack id. ex) zsef123
    """
    from slackclient import SlackClient
    slack_client = SlackClient(os.environ.get('SLACK_BOT_TOKEN'))
    if slack_client.rtm_connect(with_team_state=False):
        ret = slack_client.api_call("chat.postMessage", channel="@"+send_id, text=send_msg, as_user=True)
        resp = "Send Failed" if ret['ok'] == False else "To %s, send %s"%(send_id, send_msg)
        print(resp)
    else:
        print("Client connect Fail")

if __name__=="__main__":
    from sklearn.metrics import jaccard_similarity_score, f1_score
    y_true = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    y_pred = np.array([[1, 1, 0], [0, 0, 0], [1, 0, 0]])

    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()

    sk_jss   = jaccard_similarity_score(y_true, y_pred)
    sk_jss_f = jaccard_similarity_score(y_true_f, y_pred_f)

    confusion = confusion_matrix(y_true_f, y_pred_f).ravel()
    print(confusion)
    sensitivity, specificity, precision, recall, f1, jaccard, dice = get_roc_pr(*confusion)
    print("sk_jss : ", sk_jss, "sk_jss_f : ", sk_jss_f, "jss : ", jaccard)


