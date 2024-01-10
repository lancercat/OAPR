import torch;

from neko_2020nocr.dan.common.common import flatten_label
from neko_2021_mjt.modulars.neko_inflater import neko_inflater
from neko_2021_mjt.routines.subroutines.cores.mk7g3 import recog_loss
from neko_2022_soai_zero.routines.ocr_routine.mk7g3.osdan_routine_mk7g3_rec import \
    neko_HDOS2C_routine_CFmk7g3_rec, neko_HDOS2C_routine_CFmk7g3_rec_core, neko_fixed_cwa_subroutine


class neko_part_subroutibne:
    def fp_impl(this,fout_emb,A,proto,label_flatten,tdict,device,modular_dict,length):
        loss=torch.tensor(0,device=device).float();
        termdic={}
        if(modular_dict["part_loss"]!="NEP_skipped_NEP"):
            ovr=modular_dict["part_loss"](A,length);
            loss +=ovr;
            termdic["part_loss"]=ovr.item();
            pass;
        return loss,termdic;


class neko_HDOS2C_routine_CFmk7g3AP_rec_core(neko_HDOS2C_routine_CFmk7g3_rec_core):
    def arm_submodules(this):
        this.inflater = neko_inflater();
        this.water_mod=neko_fixed_cwa_subroutine();
        this.part_loss=neko_part_subroutibne();

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
        prototyper = modular_dict["prototyper"];
        proto = prototyper(normprotos, use_sp=False);
        label_flatten, length = flatten_label(target, EOSlen=0, length=length);
        target, label_flatten, culength = target.to(device), label_flatten.to(device), length.long().to(device)
        if(modular_dict["aam"] != "NEP_skipped_NEP"):
            aclips = modular_dict["aam"](clips.to(device));
        else:
            aclips=clips.to(device);
        out_emb, A, pred_length = this.fe_seq(aclips, modular_dict, length);
        fout_emb, _ = this.inflater.inflate(out_emb, length)


        water_loss, water_term = this.water_mod.fp_impl(fout_emb, proto, normprotos, label_flatten, tdict, device,
                                                        modular_dict);
        part_loss, part_term = this.part_loss.fp_impl(
            fout_emb,A,proto,label_flatten,tdict,device,modular_dict,length);
        cls_loss, cls_terms, beams = recog_loss(modular_dict, pred_length, culength, fout_emb, proto, plabel,
                                                label_flatten, length, tdict);

        loss = cls_loss + 0.1 * water_loss+0.1*part_loss;
        # computing accuracy and loss
        tarswunk = ["".join([tdict[i.item()] for i in target[j]]).replace('[s]', "") for j in
                    range(len(target))];

        logger_dict["accr"].add_iter(beams[0], length, tarswunk)
        logger_dict["loss"].add_iter(loss, {"cls": cls_terms, "water": water_term,"part":part_term});
        return loss;


class neko_HDOS2C_routine_CFmk7g3AP(neko_HDOS2C_routine_CFmk7g3_rec):
    def set_etc(this, args):
        this.maxT = args["maxT"];
        this.core=neko_HDOS2C_routine_CFmk7g3AP_rec_core();
class neko_HDOS2C_routine_CFmk7g3APD(neko_HDOS2C_routine_CFmk7g3AP):
    def set_etc(this, args):
        this.maxT = args["maxT"];
        this.delay_till=args["delay_till"];
        this.core=neko_HDOS2C_routine_CFmk7g3AP_rec_core();
    def fp_impl(this, input_dict, modular_dict, logger_dict, nEpoch, batch_idx, device):
        sampler=modular_dict["sampler"];
        normprotos, semb, plabel, tdict=this.mk_proto(input_dict["label"],sampler);
        target, length = sampler.model.encode_noeos(normprotos, plabel, tdict, input_dict["label"]);
        exdict={};
        exdict["proto"]=normprotos;
        exdict["target"]=target;
        exdict["length"]=length;
        exdict["plabel"]=plabel;
        exdict["tdict"]=tdict;
        exdict["semb"] = semb;
        if(nEpoch==0 and batch_idx<this.delay_till):
            modular_dict["aam"]="NEP_skipped_NEP";
            modular_dict["part_loss"]="NEP_skipped_NEP";
        loss=this.core.fp_impl(input_dict,exdict,modular_dict,logger_dict,device)
        return loss;

class neko_HDOS2C_routine_CFmk7g3APM_rec_core(neko_HDOS2C_routine_CFmk7g3_rec_core):
    def arm_submodules(this):
        this.inflater = neko_inflater();
        this.water_mod=neko_fixed_cwa_subroutine();
        this.part_loss=neko_part_subroutibne();
    def fe_seq(this,clips,modular_dict,length):
        features_ffn,features = modular_dict["feature_extractor"](clips)
        features=[f.contiguous() for f in features];
        # features_ffn=[ for f in features_ffn];
        A,RA,pred_length = modular_dict["CAM"](features);
        out_emb = modular_dict["seq"](features_ffn[-1].contiguous(), A, length);
        return out_emb,A,RA,pred_length;
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
        prototyper = modular_dict["prototyper"];
        proto = prototyper(normprotos, use_sp=False);
        label_flatten, length = flatten_label(target, EOSlen=0, length=length);
        target, label_flatten, culength = target.to(device), label_flatten.to(device), length.long().to(device)
        if(modular_dict["aam"] != "NEP_skipped_NEP"):
            aclips = modular_dict["aam"](clips.to(device));
        else:
            aclips=clips.to(device);
        out_emb, A,RA, pred_length = this.fe_seq(aclips, modular_dict, length);
        fout_emb, _ = this.inflater.inflate(out_emb, length)
        att_loss=this.attloss
        water_loss, water_term = this.water_mod.fp_impl(fout_emb, proto, normprotos, label_flatten, tdict, device,
                                                        modular_dict);
        part_loss, part_term = this.part_loss.fp_impl(fout_emb,A,RA,proto,label_flatten,tdict,device,modular_dict);
        cls_loss, cls_terms, beams = recog_loss(modular_dict, pred_length, culength, fout_emb, proto, plabel,
                                                label_flatten, length, tdict);

        loss = cls_loss + 0.1 * water_loss+0.1*part_loss;
        # computing accuracy and loss
        tarswunk = ["".join([tdict[i.item()] for i in target[j]]).replace('[s]', "") for j in
                    range(len(target))];

        logger_dict["accr"].add_iter(beams[0], length, tarswunk)
        logger_dict["loss"].add_iter(loss, {"cls": cls_terms, "water": water_term,"part":part_term});
        return loss;


class neko_HDOS2C_routine_CFmk7g3APM(neko_HDOS2C_routine_CFmk7g3_rec):
    def set_etc(this, args):
        this.maxT = args["maxT"];
        this.core=neko_HDOS2C_routine_CFmk7g3APM_rec_core();


class neko_HDOS2C_routine_CFmk7g3APT_rec_core(neko_HDOS2C_routine_CFmk7g3AP_rec_core):
    def fe_seq(this,clips,modular_dict,length):
        features_ffn,features = modular_dict["feature_extractor"](clips)
        features=[f.contiguous() for f in features];
        # features_ffn=[ for f in features_ffn];
        A,pred_length = modular_dict["CAM"](features)
        out_emb = modular_dict["seq"](features_ffn[-1].contiguous(), A, length);
        return out_emb,A,pred_length;

class neko_HDOS2C_routine_CFmk7g3APT(neko_HDOS2C_routine_CFmk7g3AP):
    def set_etc(this, args):
        this.maxT = args["maxT"];
        this.core=neko_HDOS2C_routine_CFmk7g3APT_rec_core();
