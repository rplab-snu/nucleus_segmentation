# utils.py
import numpy as np
from scipy.ndimage import zoom
from skimage.transform import iradon

def get_save_dir(arg):
    if arg.model == "fusion":
        save_path = "%s/%s_%s_ep%d_batch%d_ngf%d_lrG%d_beta%f_%f" %(
                      arg.save_dir, arg.model, arg.output,
                      arg.epoch, arg.batch_size, arg.ngf, arg.lrG, arg.beta1, arg.beta2
                     )
    elif arg.model == "pix2pix":
        save_path = "%s/%s_%s_ep%d_batch%d_ngf%d_lrG%d_lrD_beta%f_%f" % (
                      arg.save_dir, arg.model, arg.output,
                      arg.epoch, arg.batch_size, arg.ngf, arg.lrG, arg.lrD, arg.beta1, arg.beta2
                      )
    else:
        raise NotImplementedError("arg.model is %s"%(arg.model))
    return save_path


def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def _tr_to_np(tr):
    tr = tr.cpu().data[0].numpy()
    return tr.reshape(tr.shape[1], tr.shape[2])

def sin2res(input_, output_, target_ = None):
    # 725 x 512 (원래 sinogram dimension)
    input_  = zoom(input_ * 100,  (725 / 512, 1), order=1)
    output_ = zoom(output_ * 100, (725 / 512, 1), order=1)

    # Final sinogram = Inserted - Residual
    sinograms = [input_ - output_]

    fbps = [iradon(input_, theta=theta, circle=False) * 4095.0]
    fbps.append(iradon(Final_sinogram, theta=theta, circle=False) * 4095.0)
    if target is not None:
        sinograms.append(input_ - target_ * 100)
        fbps.append(iradon(Targt_sinogram, theta=theta, circle=False) * 4095.0)
    
    Total_fbp = np.concatenate(fbps,                   axis=1)
    Total_sin = np.concatenate(sinograms,              axis=1)
    Total     = np.concatenate((Total_sin, Total_fbp), axis=0)
    return Total

def image_save(self, save_path, type_, input_, output_, target_=None):
    if type_ == "img2img":
        if target_ is None:
            Total = np.concatenate((input_, output_), axis = 1)
        else:
            Total = np.concatenate((input_, target_, output_), axis = 1)

    elif type_ == "img2res":                
        # MAR_image = MA_image - Residual
        if target_ is None:
            MAR_image = (input_ - output_)  * 4095
            Total = np.concatenate((input_ * 4095.0, MAR_image), axis = 1)
        else:
            Target_image = (input_ - target_) * 4095.0
            MAR_image = (input_ - output_)  * 4095
            Total = np.concatenate((input_ * 4095.0, Target_image, MAR_image), axis = 1)

    elif type_ == "sin2res":        
        Total = utils.sin2res(input_, output_, target_)
        
    else:
        raise NotImplementedError("self.output is {}".format(type_))

    np.save(save_path +'.npy', Total)
    scipy.misc.imsave(save_path + '.png', Total)

def slack_alarm(send_id, send_msg="Train Done"):
    """
    send_id : slack id. ex) zsef123
    """
    import os
    from slackclient import SlackClient
    slack_client = SlackClient(os.environ.get('SLACK_BOT_TOKEN'))
    if slack_client.rtm_connect(with_team_state=False):
        ret = slack_client.api_call("chat.postMessage", channel="@"+send_id, text=send_msg, as_user=True)
        resp = "Send Failed" if ret['ok'] == False else "To %s, send %s"%(send_id, send_msg)
        print(resp)
    else:
        print("Client connect Fail")

