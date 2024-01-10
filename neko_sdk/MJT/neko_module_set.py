import copy
import os
import queue

from neko_sdk.MJT.bogo_module.bogo_modular import neko_bogo_modular
from neko_sdk.MJT.common import Updata_Parameters
from neko_sdk.MJT.neko_modular import neko_modular
from neko_sdk.MJT.utils import update
from neko_sdk.thirdparty.mmdetapply import multi_apply


class neko_module_set:
    # if ("provides_bogo_modules" in cfg):
    #     this.bogo_modular_dict[name] = {}
    #     for k in cfg["provides_bogo_modules"]:
    #         this.bogo_modular_dict[name][k] = mod.bogog2_modules[k];
    def replicate(this,devices):
        q=queue.Queue()
        ret={};
        for dev in devices:
            ret[dev]={};
        for name in this.modular_dict:
            try:
                mods=this.modular_dict[name].replicate(devices)
                for devid in range(len(devices)):
                    ret[devices[devid]][name]=mods[devid];
            except:
                mods = this.modular_dict[name].replicate(devices)
                for devid in range(len(devices)):
                    ret[devices[devid]][name] = mods[devid];
                print("some thing is wrong with duplicating ",name)
                q.put(name);
        while not q.empty():
            name=q.get();
            try:
                mods = this.modular_dict[name].replicate(devices)
                for devid in range(len(devices)):
                    ret[devices[devid]][name] = mods[devid];
            except:
                print("some thing is wrong with duplicating ", name)
                q.put(name);
        return ret;
    def attempt_arm_bogo_list(this,bogolist,modcfgs):
        fail_list=[];
        for name in bogolist:
            cfg = modcfgs[name];
            # bogo modules are re-combination of parts of existing modules.
            try:
                mod= cfg["bogo_mod"](cfg["args"],this.modular_dict);
                this.modular_dict[name] = neko_bogo_modular(mod);
            except:
                fail_list.append(name);
        return fail_list;


    def arm_modules(this, root, modcfgs,itrkey):
        this.optimizers = [];
        this.optnames=[];
        this.optimizer_schedulers = [];
        this.modular_dict = {};
        this.bogo_modular_list=[]
        for name in modcfgs:
            cfg = modcfgs[name];
            # so that you don't set None and forget.. You will have to explicitly skip with this string.
            if(cfg=="NEP_skipped_NEP"):
                # You have to handle the missing module...
                # We will not let you use None as you need to be explicitly aware the modules skipped.
                this.modular_dict[name] = "NEP_skipped_NEP";
                continue;
            modp=os.path.join(root, name);
            if("bogo_mod" in cfg):
                this.bogo_modular_list.append(name);
            else:
                mod, opt, opts = cfg["modular"](cfg["args"], modp, modp);
                this.modular_dict[name] = neko_modular(modp,name, mod, cfg["save_each"]);
                this.modular_dict[name].load(itrkey)
                if (opt is not None):
                    this.optimizers.append(opt);
                    this.optnames.append(name)
                    this.optimizer_schedulers.append(opts);
        list_bogo_to_arm=copy.copy(this.bogo_modular_list);
        for i in range(40):
            if(len(list_bogo_to_arm)==0):
                break;
            if(i):
                print("Attempt",i,"for",list_bogo_to_arm);
            list_bogo_to_arm=this.attempt_arm_bogo_list(list_bogo_to_arm,modcfgs);
        if(len(list_bogo_to_arm)):
            print("failed dependency for module(s):",list_bogo_to_arm,"please check dependency");
            exit(9);
        # make sure we have collected real modules.


    def eval_mode(this):
        for modk in this.modular_dict:
            if(this.modular_dict[modk]=="NEP_skipped_NEP"):
                continue;
            this.modular_dict[modk].eval();


    def zero_grad(this):
        for modk in this.modular_dict:
            if(this.modular_dict[modk]=="NEP_skipped_NEP"):
                continue;
            this.modular_dict[modk].zero_grad();
    def train_mode(this):
        for modk in this.modular_dict:
            if(this.modular_dict[modk]=="NEP_skipped_NEP"):
                continue;
            this.modular_dict[modk].train();
    def save_necessary(this,nEpoch, batch_idx):
        for modk in this.modular_dict:
            if(this.modular_dict[modk]=="NEP_skipped_NEP"):
                continue;
            this.modular_dict[modk].save_if_needed(nEpoch, batch_idx);
    def update_para(this):
        multi_apply(update, this.optimizers);
    def update(this):

        try:
            Updata_Parameters(this.optimizers, frozen=[]);
        except:
            print("Oops");
            exit(9);

    def norm_grad(this):
        for modk in this.modular_dict:
            if(this.modular_dict[modk]=="NEP_skipped_NEP"):
                continue;
            if (this.modular_dict[modk].save_each > 0):
                this.modular_dict[modk].normgrad();
