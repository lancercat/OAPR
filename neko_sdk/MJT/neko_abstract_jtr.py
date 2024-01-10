import os
import random
import time

import numpy
import torch
from torch.nn.parallel import parallel_apply

from neko_sdk.MJT.common import Updata_Parameters
from neko_sdk.MJT.neko_module_set import neko_module_set
from neko_sdk.MJT.utils import update
from neko_sdk.thirdparty.mmdetapply import multi_apply


class neko_abstract_modular_joint_training(neko_module_set):

    def set_routines(this,routine_cfgs):
        this.routines=[];
        this.routine_names=[]
        for rcfg in routine_cfgs:
            this.routine_names.append(rcfg);
            this.routines.append(routine_cfgs[rcfg]["routine"](routine_cfgs[rcfg]))

    def set_val_tasks(this,val_cfgs):
        this.val_tasks = [];
        for vk in val_cfgs:
            this.val_tasks.append(val_cfgs[vk]["type"](None,None,this.modular_dict,val_cfgs[vk],1000))
    def set_dataloader(this,datacfg,vitr):
        this.joint_dataloader=datacfg["loadertype"](datacfg,vitr);
    def setup(this,cfgs):
        root, this.val_each, this.vitr, this.vepoch = \
            cfgs["root"], cfgs["val_each"], cfgs["vitr"], cfgs["vepoch"];
        # set to "latest" for resuming, whatever does not make sense to start fresh.
        this.set_dataloader(cfgs["dataloader_cfg"], vitr=cfgs["vitr"]);
        this.arm_modules(root, cfgs["modules"], cfgs["iterkey"]);
        this.set_routines(cfgs["routine_cfgs"]);
        this.set_val_tasks(cfgs["tasks"]);

    def __init__(this,
                 cfgs):
        seed = 9;
        torch.manual_seed(seed);
        torch.cuda.manual_seed_all(seed);
        torch.cuda.manual_seed(seed);
        numpy.random.seed(seed);
        random.seed(seed);
        print("We are running from commit,",os.popen('git rev-parse HEAD').read())
        this.setup(cfgs)
        pass;

        # ---------------------------------
    def val(this,nEpoch,batch_idx,vdbg=None):
        this.eval_mode()
        # torch.cuda.empty_cache();
        for vt in this.val_tasks:
            print(nEpoch,batch_idx);
            torch.cuda.empty_cache();
            with torch.no_grad():
                vt.test(vdbg=vdbg);
        torch.cuda.empty_cache();
        this.train_mode();
    def launch(this,rot,sample_batched,nEpoch,batch_idx):
        rot.fpbp(sample_batched, this.modular_dict, nEpoch, batch_idx)
        return []
    def tr_iter_amp(this,nEpoch,batch_idx,sample_batched):
        # torch.autograd.set_detect_anomaly(True);
        # data prepare
        zg_start=time.time();
        for modk in this.modular_dict:
            if(this.modular_dict[modk]=="NEP_skipped_NEP"):
                continue;
            this.modular_dict[modk].zero_grad();

        routine_start=time.time();
        # multi_apply(this.launch,this.routines, sample_batched=sample_batched, nEpoch=nEpoch,
        #             batch_idx=batch_idx)
        #
        for routine in this.routines:
            routine.fpbp_amp(sample_batched,this.modular_dict,nEpoch,batch_idx)
        # reqnorm=[]
        # for modk in this.modular_dict:
        #     if(this.modular_dict[modk].save_each>0):
        #         reqnorm.append(this.modular_dict[modk]);
        # multi_apply(normgrad,reqnorm)

        for modk in this.modular_dict:
            if(this.modular_dict[modk]=="NEP_skipped_NEP"):
                continue;
            if(this.modular_dict[modk].save_each>0):
                ng_start_ = time.time();
                this.modular_dict[modk].normgrad();
                # print(modk,time.time()-ng_start_);
        pu_start=time.time();
        try:
            Updata_Parameters(this.optimizers, frozen=[]);
        except:
            print("Oops");
            exit(9);
        all_done=time.time();

        if(batch_idx%100==9):
            print("[Timings]: zg:",routine_start-zg_start, "routines:", pu_start-routine_start,"pu:",all_done-pu_start);

        for modk in this.modular_dict:
            if(this.modular_dict[modk]=="NEP_skipped_NEP"):
                continue;
            this.modular_dict[modk].save_if_needed(nEpoch,batch_idx);

    def tr_iter(this,nEpoch,batch_idx,sample_batched):
        # torch.autograd.set_detect_anomaly(True);
        # data prepare
        zg_start=time.time();
        for modk in this.modular_dict:
            if(this.modular_dict[modk]=="NEP_skipped_NEP"):
                continue;
            this.modular_dict[modk].zero_grad();

        routine_start=time.time();
        # multi_apply(this.launch,this.routines, sample_batched=sample_batched, nEpoch=nEpoch,
        #             batch_idx=batch_idx)
        #
        for routine in this.routines:
            routine.fpbp(sample_batched,this.modular_dict,nEpoch,batch_idx)
        # reqnorm=[]
        # for modk in this.modular_dict:
        #     if(this.modular_dict[modk].save_each>0):
        #         reqnorm.append(this.modular_dict[modk]);
        # multi_apply(normgrad,reqnorm)

        for modk in this.modular_dict:
            if(this.modular_dict[modk]=="NEP_skipped_NEP"):
                continue;
            if(this.modular_dict[modk].save_each>0):
                ng_start_ = time.time();
                this.modular_dict[modk].normgrad();
                # print(modk,time.time()-ng_start_);
        pu_start=time.time();
        try:
            Updata_Parameters(this.optimizers, frozen=[]);
        except:
            print("Oops");
            exit(9);
        all_done=time.time();

        if(batch_idx%100==9):
            print("[Timings]: zg:",routine_start-zg_start, "routines:", pu_start-routine_start,"pu:",all_done-pu_start);

        for modk in this.modular_dict:
            if(this.modular_dict[modk]=="NEP_skipped_NEP"):
                continue;
            this.modular_dict[modk].save_if_needed(nEpoch,batch_idx);


    def train(this,dbgpath,vdbg=None,flag=None):
        torch.backends.cudnn.benchmark=True;

        for modk in this.modular_dict:
            if(this.modular_dict[modk] == "NEP_skipped_NEP"):
                continue;
            this.modular_dict[modk].cuda();
            this.modular_dict[modk].train();
        for nEpoch in range(0, this.vepoch):
            for batch_idx in range(this.vitr):
                if(flag is None or flag==False):
                    flag = (batch_idx > 0) or (dbgpath is not None);
                if (flag and batch_idx % this.val_each == 0):
                    this.val(nEpoch, batch_idx,vdbg=vdbg);
                data_start=time.time();
                sample_batched=this.joint_dataloader.next();
                data_time=time.time()-data_start;
                itr_start=time.time();
                if(dbgpath is not None):
                    sample_batched["debug_path"]=dbgpath;
                if(vdbg is not None):
                    sample_batched["vdbg"]=vdbg;

                # for d in sample_batched:
                #     if(type(sample_batched[d])==torch.tensor):
                #         sample_batched[d]=sample_batched[d].cuda()
                this.tr_iter(nEpoch,batch_idx,sample_batched)
                itr_time = time.time()-itr_start;

                # print(torch.backends.cudnn.benchmark);
                if(batch_idx%100==9):
                    print("datatime",data_time,"itrtime",itr_time,"all",time.time()-data_start);
            Updata_Parameters(this.optimizer_schedulers, frozen=[])
            this.val(nEpoch, "Final");

            # torch.backends.cudnn.benchmark = False;
            for modk in this.modular_dict:
                if (this.modular_dict[modk] == "NEP_skipped_NEP"):
                    continue;
                this.modular_dict[modk].save(nEpoch);
class neko_abstract_modular_joint_eval(neko_module_set):

    def set_val_tasks(this,val_cfgs,mitr):
        this.val_tasks = [];
        this.val_keys=[];
        for vk in val_cfgs:
            this.val_keys.append(vk);
            this.val_tasks.append(val_cfgs[vk]["type"](None,None,this.modular_dict,val_cfgs[vk],mitr))
    def test_img(this,id,image_path,globalcache,h=32,w=100):
        return this.val_tasks[id].test_image(image_path,globalcache)

    def test_img_top_k(this, id, image_path,attover_paths, globalcache, h=32, w=100):
        return this.val_tasks[id].test_top_k(image_path,attover_paths, globalcache)

    def pretest(this,id):
        this.eval_mode();
        return this.val_tasks[id].testready();

    def __init__(this,
                 cfgs,mitr):
        root= \
        cfgs["root"];
        # set to "latest" for resuming, whatever does not make sense to start fresh.
        this.arm_modules(root,cfgs["modules"],cfgs["iterkey"]);
        for mk in this.modular_dict:
            if(this.modular_dict[mk]=="NEP_skipped_NEP"):
                continue;
            this.modular_dict[mk].model.cuda();
        if("export_path" in cfgs and cfgs["export_path"] is not None):
            for k in cfgs["tasks"]:
                cfgs["tasks"][k]["export_path"]=cfgs["export_path"];

        this.set_val_tasks(cfgs["tasks"],mitr);
        pass;

        # ---------------------------------
    def val(this,nEpoch,batch_idx,rot=0,vdbg=None):
        this.eval_mode();
        tasklogs={};
        for vid in range(len(this.val_tasks)):
            print(this.val_keys[vid],nEpoch,batch_idx,"Starts","------------------------");
            torch.cuda.empty_cache();
            with torch.no_grad():
                tasklogs[vid]=this.val_tasks[vid].test(rot,logname="E"+str(nEpoch)+"_I"+str(batch_idx),vdbg=vdbg);
            print("------------------------------------------------------");

        this.train_mode()

    def vis(this, nEpoch, batch_idx, rot=0):
        this.eval_mode()
        for vt in this.val_tasks:
            print(nEpoch, batch_idx);
            torch.cuda.empty_cache();
            vt.visualize(rot);
        this.train_mode()

        # ---------------------------------
    def valt(this,nEpoch,batch_idx):
        this.train_mode()
        for vt in this.val_tasks:
            print(nEpoch,batch_idx);
            with torch.no_grad():
                torch.cuda.empty_cache()
                vt.test();
        this.train_mode()
# some routines may change shared module state.
class neko_modular_joint_training_semipara(neko_abstract_modular_joint_training):
    def tr_iter(this,nEpoch,batch_idx,sample_batched):
        # torch.autograd.set_detect_anomaly(True);
        # data prepare
        # torch.backends.cudnn.benchmark=True;

        zg_start=time.time();

        for modk in this.modular_dict:
            if(this.modular_dict[modk]=="NEP_skipped_NEP"):
                continue;
            this.modular_dict[modk].zero_grad();
        routine_start=time.time();

        # multi_apply(this.launch,this.routines, sample_batched=sample_batched, nEpoch=nEpoch,
        #             batch_idx=batch_idx)

        # i=0;
        for routine in this.routines:
            # rs = time.time();
            routine.fpbp(sample_batched,this.modular_dict,nEpoch,batch_idx)
        #
        #     # print(this.routine_names[i],time.time()-rs);
        #     # i += 1;
        ng_start=time.time();

        for modk in this.modular_dict:
            if(this.modular_dict[modk]=="NEP_skipped_NEP"):
                continue;
            if(this.modular_dict[modk].save_each>0):
                this.modular_dict[modk].normgrad();
        pu_start=time.time();
        multi_apply(update,this.optimizers);
        # try:
        #     Updata_Parametersd(this.optimizers,this.optnames, frozen=[]);
        # except:
        #     print("Oops");
        #     exit(9);
        all_done=time.time();
        if (batch_idx % 100 == 9):
            print("[Timings]: zg:", routine_start - zg_start, "routines:", pu_start - routine_start, "pu:", all_done - pu_start);

        for modk in this.modular_dict:
            if(this.modular_dict[modk]=="NEP_skipped_NEP"):
                continue;
            this.modular_dict[modk].save_if_needed(nEpoch,batch_idx);

class neko_modular_joint_training_para(neko_abstract_modular_joint_training):
    def tr_iter(this,nEpoch,batch_idx,sample_batched):
        # torch.autograd.set_detect_anomaly(True);
        # data prepare
        # torch.backends.cudnn.benchmark=True;

        zg_start=time.time();

        for modk in this.modular_dict:
            if(this.modular_dict[modk]=="NEP_skipped_NEP"):
                continue;
            this.modular_dict[modk].zero_grad();
        routine_start=time.time();

        multi_apply(this.launch,this.routines, sample_batched=sample_batched, nEpoch=nEpoch,
                    batch_idx=batch_idx)

        # i=0;
        # for routine in this.routines:
        #     # rs = time.time();
        #     routine.fpbp(sample_batched,this.modular_dict,nEpoch,batch_idx)
        #
        #     # print(this.routine_names[i],time.time()-rs);
        #     # i += 1;
        ng_start=time.time();

        # for modk in this.modular_dict:
        #     if(this.modular_dict[modk].save_each>0):
        #         this.modular_dict[modk].normgrad();
        pu_start=time.time();
        multi_apply(update,this.optimizers);
        # try:
        #     Updata_Parameters(this.optimizers, frozen=[]);
        # except:
        #     print("Oops");
        #     exit(9);
        all_done=time.time();
        if (batch_idx % 100 == 9):
            print("[Timings]: zg:", routine_start - zg_start, "routines:", pu_start - routine_start, "pu:", all_done - pu_start);

        for modk in this.modular_dict:
            if(this.modular_dict[modk]=="NEP_skipped_NEP"):
                continue;
            this.modular_dict[modk].save_if_needed(nEpoch,batch_idx);

class neko_modular_joint_training_para2(neko_abstract_modular_joint_training):
    def launch(this,rot,sample_batched,nEpoch,batch_idx):
        l= rot.fp(sample_batched, this.modular_dict, nEpoch, batch_idx, "cuda")
        return [l]
    def tr_iter(this,nEpoch,batch_idx,sample_batched):
        # torch.autograd.set_detect_anomaly(True);
        # data prepare
        # torch.backends.cudnn.benchmark=True;

        zg_start=time.time();

        for modk in this.modular_dict:
            this.modular_dict[modk].zero_grad();
        routine_start=time.time();

        losses=multi_apply(this.launch,this.routines, sample_batched=sample_batched, nEpoch=nEpoch,
                    batch_idx=batch_idx)
        loss=torch.stack([loss[0] for loss in losses]).sum();

        #
        # loss=0;
        # for routine in this.routines:
        #     # rs = time.time();
        #     loss+=routine.fp(sample_batched,this.modular_dict,nEpoch,batch_idx)
        #     # print(this.routine_names[i],time.time()-rs);
        #     # i += 1;
        #
        loss.backward();
        ng_start=time.time();

        for modk in this.modular_dict:
            if(this.modular_dict[modk].save_each>0):
                this.modular_dict[modk].normgrad();
        pu_start=time.time();
        # multi_apply(update,this.optimizers);
        try:
            Updata_Parameters(this.optimizers, frozen=[]);
        except:
            print("Oops");
        #     exit(9);
        all_done=time.time();
        if (batch_idx % 100 == 9):
            print("[Timings]: zg:", routine_start - zg_start, "routines:", pu_start - routine_start, "pu:", all_done - pu_start);

        for modk in this.modular_dict:
            this.modular_dict[modk].save_if_needed(nEpoch,batch_idx);


class neko_modular_joint_training_para3(neko_abstract_modular_joint_training):
    def launch(this,rot,sample_batched,nEpoch,batch_idx):
        l= rot.fp(sample_batched, this.modular_dict, nEpoch, batch_idx, "cuda")
        return [l]
    def tr_iter(this,nEpoch,batch_idx,sample_batched):
        # torch.autograd.set_detect_anomaly(True);
        # data prepare
        # torch.backends.cudnn.benchmark=True;

        zg_start=time.time();

        for modk in this.modular_dict:
            this.modular_dict[modk].zero_grad();
        routine_start=time.time();
        inp=[[sample_batched,this.modular_dict,nEpoch,batch_idx] for _ in this.routines]
        dev=["cuda" for _ in this.routines];
        parallel_apply(this.routines,inp,devices=dev)

        #
        # loss=0;
        # for routine in this.routines:
        #     # rs = time.time();
        #     loss+=routine.fp(sample_batched,this.modular_dict,nEpoch,batch_idx)
        #     # print(this.routine_names[i],time.time()-rs);
        #     # i += 1;
        #
        # loss.backward();
        ng_start=time.time();

        for modk in this.modular_dict:
            if(this.modular_dict[modk].save_each>0):
                this.modular_dict[modk].normgrad();
        pu_start=time.time();
        # multi_apply(update,this.optimizers);
        try:
            Updata_Parameters(this.optimizers, frozen=[]);
        except:
            print("Oops");
        #     exit(9);
        all_done=time.time();
        if (batch_idx % 100 == 9):
            print("[Timings]: zg:", routine_start - zg_start, "routines:", pu_start - routine_start, "pu:", all_done - pu_start);

        for modk in this.modular_dict:
            if(this.modular_dict[modk]=="NEP_skipped_NEP"):
                continue;
            this.modular_dict[modk].save_if_needed(nEpoch,batch_idx);
