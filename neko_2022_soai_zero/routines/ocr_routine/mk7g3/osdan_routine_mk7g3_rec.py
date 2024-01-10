import torch
from torch.nn import functional as trnf

from neko_2020nocr.dan.common.common import flatten_label
from neko_2021_mjt.modulars.neko_inflater import neko_inflater
from neko_2021_mjt.routines.ocr_routines.mk7g3.osdan_routine_mk7g3 import neko_HDOS2C_routine_CFmk7g3
from neko_2021_mjt.routines.subroutines.cores.mk7g3 import neko_HDOS2C_routine_CFmk7g3_core, recog_loss


class neko_cwa_subroutine:
    def fp_impl(this,fout_emb,proto,normprotos,label_flatten,tdict,device,modular_dict):
        loss=torch.tensor(0,device=device).float();
        terms={};
        if(modular_dict["dom_mix"]!="NEP_skipped_NEP"):
            l_dommix=modular_dict["dom_mix"]([proto,fout_emb]);
            loss += l_dommix;
            terms["dom_mix"] = l_dommix.item();

        if(modular_dict["p_recon"]!="NEP_skipped_NEP"):
            l_precon=modular_dict["p_recon"](proto.unsqueeze(-1).unsqueeze(-1),torch.cat(normprotos));
            loss+=l_precon;
            terms["p_recon"]=l_precon.item();

        return loss,terms;
class neko_fixed_cwa_subroutine:
    def fp_impl(this,fout_emb,proto,normprotos,label_flatten,tdict,device,modular_dict):
        loss=torch.tensor(0,device=device).float();
        normed_fout=trnf.normalize(fout_emb,dim=-1,p=2);

        terms={};
        if(modular_dict["dom_mix"]!="NEP_skipped_NEP"):
            l_dommix=modular_dict["dom_mix"]([proto,normed_fout]);
            loss += l_dommix;
            terms["dom_mix"] = l_dommix.item();

        if(modular_dict["p_recon"]!="NEP_skipped_NEP"):
            l_precon=modular_dict["p_recon"](proto.unsqueeze(-1).unsqueeze(-1),torch.cat(normprotos));
            loss+=l_precon;
            terms["p_recon"]=l_precon.item();

        if( modular_dict["f_recon"]!="NEP_skipped_NEP"):
            l_frecon=modular_dict["f_recon"](normed_fout.unsqeeze(-1).unsqueeze(-1),normprotos,tdict,label_flatten);
            loss +=  l_frecon;
            terms["f_recon"]= l_frecon.item();
        return loss,terms;

class neko_HDOS2C_routine_CFmk7g3_rec_core(neko_HDOS2C_routine_CFmk7g3_core):
    def arm_submodules(this):
        this.inflater = neko_inflater();
        this.water_mod=neko_cwa_subroutine();

    def fp_impl(this, input_dict, exdict, modular_dict, logger_dict, device):
        clips = input_dict["image"];

        # Prototypes(sampled)
        # And this helps using SYNTH words in LSCT
        target = exdict["target"];
        length = exdict["length"];
        tdict = exdict["tdict"];
        normprotos = exdict["proto"];
        # semb=exdict["semb"];
        plabel = exdict["plabel"];

        prototyper = modular_dict["prototyper"]

        proto = prototyper(normprotos, use_sp=False);
        label_flatten, length = flatten_label(target, EOSlen=0, length=length);
        target, label_flatten, culength = target.to(device), label_flatten.to(device), length.long().to(device)
        out_emb, A, pred_length = this.fe_seq(clips.to(device), modular_dict, length);
        fout_emb, _ = this.inflater.inflate(out_emb, length)

        water_loss, water_term = this.water_mod.fp_impl(fout_emb, proto, normprotos, label_flatten, tdict, device,
                                                        modular_dict);
        cls_loss, cls_terms, beams = recog_loss(modular_dict, pred_length, culength, fout_emb, proto, plabel,
                                                label_flatten, length, tdict);

        loss = cls_loss + 0.1 * water_loss;
        # computing accuracy and loss
        tarswunk = ["".join([tdict[i.item()] for i in target[j]]).replace('[s]', "") for j in
                    range(len(target))];

        logger_dict["accr"].add_iter(beams[0], length, tarswunk)
        logger_dict["loss"].add_iter(loss, {"cls": cls_terms, "water": water_term})
        return loss;


# mk5 CF branch dropped predict-sample-predict support.
# A GP branch will be added if it's ever to be supported
# Mk7 CF branch uses CAM to perform length prediction, [s] is no more needed
# Still training only.
# It does not try to make prototype, which makes the lsct more idiot-proof
# G3 cores are for quick recasting
class neko_HDOS2C_routine_CFmk7g3_fix_rec_core(neko_HDOS2C_routine_CFmk7g3_rec_core):
    def arm_submodules(this):
        this.inflater = neko_inflater();
        this.water_mod=neko_fixed_cwa_subroutine();

class neko_HDOS2C_routine_CFmk7g3_rec(neko_HDOS2C_routine_CFmk7g3):
    def set_etc(this, args):
        this.maxT = args["maxT"];
        this.core=neko_HDOS2C_routine_CFmk7g3_rec_core();





class neko_HDOS2C_routine_CFmk7g3_fix_rec(neko_HDOS2C_routine_CFmk7g3):
    def set_etc(this, args):
        this.maxT = args["maxT"];
        # added normalization to fout_emb
        this.core=neko_HDOS2C_routine_CFmk7g3_fix_rec_core();
