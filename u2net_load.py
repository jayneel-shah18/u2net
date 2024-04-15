import os
import torch
from ai_virtual_wardrobe.u2net.model import U2NET # full size version 173.6 MB
from ai_virtual_wardrobe.u2net.model import U2NETP # small version u2net 4.7 MB


def model(model_name='u2net'):


    model_dir = 'C:\\Users\\jayne\\OneDrive\\Desktop\\Project\\Website\\ai_virtual_wardrobe\\u2net\\saved_models\\u2netp\\u2netp.pth'

    if(model_name=='u2net'):
        print("...load U2NET---173.6 MB")
        net = U2NET(3,1)
    elif(model_name=='u2netp'):
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3,1)
    net.load_state_dict(torch.load(model_dir))

    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    return net
