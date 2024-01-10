
import torch
from torch import nn
from torch.nn import parallel as trnp

from neko_sdk.MJT.bogo_module.servant_module import neko_stand_basic


class neko_modular:
    def __init__(this,path,name,module,save_each=20000):
        this.path=path;
        this.model=module;
        this.name=name;
        this.save_each=save_each;
        this.stands=None
    def get_torch_modular_dict(this):
        if(isinstance(this.model,nn.Module)):
            return this.model;
        else:
            return None;
    def replicate(this,devices):
        this.model.to(devices[0]);
        models=trnp.replicate(this.model,devices);
        this.stands= [neko_stand_basic(model) for model in models];
        return this.stands;

    def detach(this):
        this.model.requires_grad_(False)
    def attach(this):
        this.model.requires_grad_(True)

    def train(this,training=True):
        this.model.train(training);
    def eval(this):
        this.model.eval();
    def normgrad(this):
        if this.save_each>0:
            nn.utils.clip_grad_norm_(this.model.parameters(), 20, 2)

    def cuda(this):
        this.model.cuda();

    def zero_grad(this):
        if this.save_each > 0:
            for param in this.model.parameters():
                param.grad = None

            if(this.stands is not None):
                for stand in this.stands:
                    stand.model.zero_grad();


    def load(this,itrkey):
        p = this.path + itrkey + ".pth";
        try:
            this.model.load_state_dict(torch.load(p).state_dict())
        except:
            try:
                this.model.load_state_dict(torch.load(p));
                print(this.name, "loaded as a hack");
            except:
                print(this.name, "cannot load", "itr",p,", starting fresh")

    def save(this,nEpoch):
        if(this.save_each>0 ):
            torch.save(this.model, this.path+'_E{}.pth'.format(nEpoch));
            torch.save(this.model, this.path + 'latest.pth');

    def save_if_needed(this,nEpoch,batch_idx):
        if(this.save_each>0 and batch_idx%this.save_each==0):
            print("Saving", this.path + '_E{}_I{}.pth'.format(nEpoch, batch_idx))
            torch.save(this.model,this.path+'_E{}_I{}.pth'.format(nEpoch,batch_idx));
            torch.save(this.model, this.path + 'latest.pth');

    def __call__(this, *args, **kwargs):
        return this.model(*args,**kwargs);

