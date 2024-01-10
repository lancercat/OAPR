
import torch
import torch_scatter
from torch import nn
from torch.nn import functional as trnf


class osdanloss(nn.Module):
    def __init__(this, cfgs):
        super(osdanloss, this).__init__();
        this.setuploss(cfgs);

    def label_weight(this, shape, label):
        weight = torch.zeros(shape).to(label.device) + 0.1
        weight = torch.scatter_add(weight, 0, label, torch.ones_like(label).float());
        weight = 1. / weight;
        weight[-1] /= 200;
        return weight;

    def setuploss(this, cfgs):
        # this.aceloss=
        this.url = neko_unknown_ranking_loss();
        this.cosloss = neko_cos_loss2().cuda();
        this.wcls = cfgs["wcls"];
        this.wsim = cfgs["wsim"];
        this.wemb = cfgs["wemb"];
        this.wmar = cfgs["wmar"];

    def forward(this, proto, outcls, outcos, label_flatten):
        proto_loss = trnf.relu(proto[1:].matmul(proto[1:].T) - 0.14).mean();
        w = torch.ones_like(torch.ones(outcls.shape[-1])).to(proto.device).float();
        w[-1] = 0.1;
        # change introduced with va5. Masked timestamp does not contribute to loss.
        # Though we cannot say it's unknown(if the image contains one single character) --- perhaps we can?
        clsloss = trnf.cross_entropy(outcls, label_flatten, w,ignore_index=-1);
        if(this.wmar>0):
            margin_loss = this.url.forward(outcls, label_flatten, 0.5)
        else:
            margin_loss=0;

        if(outcos is not None and this.wsim>0):
            cos_loss = this.cosloss(outcos, label_flatten);
            # ace_loss=this.aceloss(outcls,label_flatten)
            loss = cos_loss * this.wsim + clsloss * this.wcls + margin_loss * this.wmar + this.wemb * proto_loss;
            terms = {
                "total": loss.detach().item(),
                "margin": margin_loss.detach().item(),
                "main": clsloss.detach().item(),
                "sim": cos_loss.detach().item(),
                "emb": proto_loss.detach().item(),
            }
        else:
            loss =  clsloss * this.wcls + margin_loss * this.wmar + this.wemb * proto_loss;
            terms = {
                "total": loss.detach().item(),
                # "margin": margin_loss.detach().item(),
                "main": clsloss.detach().item(),
                "emb": proto_loss.detach().item(),
            }
        return loss, terms
class osdanloss_clsemb(nn.Module):
    def __init__(this, cfgs):
        super(osdanloss_clsemb, this).__init__();
        this.setuploss(cfgs);

    def setuploss(this, cfgs):
        this.criterion_CE = nn.CrossEntropyLoss();
        # this.aceloss=
        this.wcls = cfgs["wcls"];
        this.wemb = cfgs["wemb"];
        this.wrej=cfgs["wrej"];
        this.reduction=cfgs["reduction"];

    def forward(this, proto, outcls, label_flatten):
        if(this.wemb>0):
            proto_loss = trnf.relu(proto[1:].matmul(proto[1:].T) - 0.14).mean();
        else:
            proto_loss=torch.tensor(0).float().to(outcls.device);
        w = torch.ones_like(torch.ones(outcls.shape[-1])).to(outcls.device).float();

        # w[-1] = 0.1;
        clsloss = trnf.cross_entropy(outcls, label_flatten, w,ignore_index=-1);
        loss =  clsloss * this.wcls  + this.wemb * proto_loss;
        terms = {
            "total": loss.detach().item(),
            "main": clsloss.detach().item(),
            "emb": proto_loss.detach().item(),
        }
        return loss, terms