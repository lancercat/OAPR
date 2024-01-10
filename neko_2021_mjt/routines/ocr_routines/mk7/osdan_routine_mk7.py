
# using mk5 DTD
import regex;
import torch
import torch.nn.functional as trnf

from neko_2020nocr.dan.common.common import flatten_label
from neko_2020nocr.dan.utils import Loss_counter, neko_os_Attention_AR_counter, neko_oswr_Attention_AR_counter
from neko_2021_mjt.modulars.neko_inflater import neko_inflater
from neko_2021_mjt.routines.ocr_routines.mk5.osdan_routine_mk5 import neko_HDOS2C_routine_CFmk5
from neko_2021_mjt.routines.subroutines.fe_seq.GTA import dump_att_ims
from neko_sdk.MJT.neko_abstract_routines import neko_abstract_eval_routine;


# mk5 CF branch dropped predict-sample-predict support.
# A GP branch will be added if it's ever to be supported
# Mk7 CF branch uses CAM to perform length prediction, [s] is no more needed
class neko_HDOS2C_routine_CFmk7(neko_HDOS2C_routine_CFmk5):

    def mk_proto(this,label,sampler,prototyper):
        normprotos, plabel, tdict=sampler.model.sample_charset_by_text(label,use_sp=False)
        # im=(torch.cat(normprotos,3)*127+128)[0][0].numpy().astype(np.uint8);
        # cv2.imshow("alphabets",im);
        # print([tdict[label.item()] for label in plabel]);
        # cv2.waitKey(0);
        proto=prototyper(normprotos,use_sp=False);

        semb=None
        return proto, semb, plabel, tdict
    def fe_seq(this,clips,modular_dict,length):
        seq=modular_dict["seq"];
        features = modular_dict["feature_extractor"](clips)
        features=[f.contiguous() for f in features];

        A,pred_length = modular_dict["CAM"](features)
        out_emb = seq(features[-1], A, length);
        return out_emb,A,pred_length;

    def fp_impl(this, input_dict, modular_dict, logger_dict, nEpoch, batch_idx, device):
        label=input_dict["label"];
        clips=input_dict["image"];
        prototyper=modular_dict["prototyper"]
        sampler=modular_dict["sampler"];
        preds=modular_dict["preds"];

        proto, semb, plabel, tdict = this.mk_proto(label,sampler,prototyper);
        target, length = sampler.model.encode_noeos(proto, plabel, tdict, label);
        label_flatten, length = flatten_label(target,EOSlen=0,length=length);
        target, label_flatten,culength = target.to(device), label_flatten.to(device),length.long().to(device)
        out_emb,A,pred_length=this.fe_seq(clips.to(device),modular_dict,length)
        # net forward
        # Length dring training is known.
        fout_emb,_=this.inflater.inflate(out_emb,length)
        lossess=[]
        beams=[];
        probs=[];
        terms=[];
        loss=trnf.cross_entropy(pred_length,culength,ignore_index=-1);
        for i in range(len(preds)):
            logits=preds[i](fout_emb, proto,plabel);
            choutput, prdt_prob = sampler.model.decode(logits, length, proto, plabel, tdict);
            loss_, terms_ = modular_dict["losses"][i](proto, logits, label_flatten);
            loss=loss_+loss;
            beams.append(choutput);
            terms.append(terms_);

        # computing accuracy and loss
        tarswunk = ["".join([tdict[i.item()] for i in target[j]]).replace('[s]', "") for j in
                    range(len(target))];
        logger_dict["accr"].add_iter(beams[0], length, tarswunk)
        logger_dict["loss"].add_iter(loss, terms[0])
        return loss;
class neko_HDOS2C_routine_CFmk7m(neko_HDOS2C_routine_CFmk7):

    # def mk_proto(this,label,sampler,prototyper):
    #     normprotos, plabel, tdict=sampler.model.sample_charset_by_text(label,use_sp=False)
    #     # im=(torch.cat(normprotos,3)*127+128)[0][0].numpy().astype(np.uint8);
    #     # cv2.imshow("alphabets",im);
    #     # print([tdict[label.item()] for label in plabel]);
    #     # cv2.waitKey(0);
    #     proto=prototyper(normprotos,use_sp=False);
    #
    #     semb=None
    #     return proto, semb, plabel, tdict
    def fe_seq(this,clips,modular_dict,length,mask):
        seq=modular_dict["seq"];
        features = modular_dict["feature_extractor"](clips.cuda())
        features=[f.contiguous() for f in features];

        A,pred_length = modular_dict["CAM"](features);
        mask=mask.float();
        if(mask.shape[-1] != A.shape[-1]):
            mask=trnf.interpolate(mask,(A.shape[-2],A.shape[-1]));
        A=A*mask.cuda();
        out_emb = seq(features[-1], A, length);
        return out_emb,A,pred_length;

    def fp_impl(this, input_dict, modular_dict, logger_dict, nEpoch, batch_idx, device):
        label=input_dict["label"];
        clips=input_dict["image"];
        prototyper=modular_dict["prototyper"]
        sampler=modular_dict["sampler"];
        preds=modular_dict["preds"];

        proto, semb, plabel, tdict = this.mk_proto(label,sampler,prototyper);
        target, length = sampler.model.encode_noeos(proto, plabel, tdict, label);
        label_flatten, length = flatten_label(target,EOSlen=0,length=length);
        target, label_flatten,culength = target.to(device), label_flatten.to(device),length.to(device).long()
        out_emb,A,pred_length=this.fe_seq(clips.to(device),modular_dict,length,input_dict["mask"]);
        # net forward
        # Length dring training is known.
        fout_emb,_=this.inflater.inflate(out_emb,length)
        lossess=[]
        beams=[];
        probs=[];
        terms=[];
        loss=trnf.cross_entropy(pred_length,culength,ignore_index=-1);
        for i in range(len(preds)):
            logits=preds[i](fout_emb, proto,plabel);
            choutput, prdt_prob = sampler.model.decode(logits, length, proto, plabel, tdict);
            loss_, terms_ = modular_dict["losses"][i](proto, logits, label_flatten);
            loss=loss_+loss;
            beams.append(choutput);
            terms.append(terms_);

        # computing accuracy and loss
        tarswunk = ["".join([tdict[i.item()] for i in target[j]]).replace('[s]', "") for j in
                    range(len(target))];
        logger_dict["accr"].add_iter(beams[0], length, tarswunk)
        logger_dict["loss"].add_iter(loss, terms[0])
        return loss;

class neko_HDOS2C_routine_CFmk7m2(neko_HDOS2C_routine_CFmk7m):

    # def mk_proto(this,label,sampler,prototyper):
    #     normprotos, plabel, tdict=sampler.model.sample_charset_by_text(label,use_sp=False)
    #     # im=(torch.cat(normprotos,3)*127+128)[0][0].numpy().astype(np.uint8);
    #     # cv2.imshow("alphabets",im);
    #     # print([tdict[label.item()] for label in plabel]);
    #     # cv2.waitKey(0);
    #     proto=prototyper(normprotos,use_sp=False);
    #
    #     semb=None
    #     return proto, semb, plabel, tdict
    def fe_seq(this,clips,modular_dict,length,mask):
        seq=modular_dict["seq"];
        mask=mask.float().cuda();
        features = modular_dict["feature_extractor"](clips.cuda())
        features=[f.contiguous() for f in features];

        A,pred_length = modular_dict["CAM"](features,mask);
        if(mask.shape[-1] != A.shape[-1]):
            mask=trnf.interpolate(mask,(A.shape[-2],A.shape[-1]));
        A=A*mask;
        out_emb = seq(features[-1], A, length);
        return out_emb,A,pred_length;

class neko_HDOS2C_routine_CFmk7dt(neko_HDOS2C_routine_CFmk7):

    def fe_seq(this,clips,modular_dict,length):
        seq=modular_dict["seq"];
        features = modular_dict["feature_extractor"](clips.cuda())
        features=[f.contiguous() for f in features];
        A,pred_length = modular_dict["CAM"](features)
        out_emb = seq(features[-1], A, length);
        return out_emb,A,pred_length;

class neko_HDOS2C_routine_CFmk7dtf(neko_HDOS2C_routine_CFmk7):

    def fe_seq(this,clips,modular_dict,length):
        seq=modular_dict["seq"];
        features = modular_dict["feature_extractor"](clips.cuda())
        features=[f.contiguous() for f in features];
        featuresd=[f.detach() for f in features];
        A,pred_length = modular_dict["CAM"](featuresd)
        out_emb = seq(features[-1], A, length);
        return out_emb,A,pred_length;

class neko_HDOS2C_eval_routine_CFmk7(neko_abstract_eval_routine):
    def dump_pred_raw(this,out_emb,modular_dict,proto,pred_length):
        nT, nB = out_emb.shape[0], out_emb.shape[1];
        preds=modular_dict["preds"];
        flogits = preds[0](out_emb.reshape([nT * nB] +list(out_emb.shape[2:])), proto, None).reshape([nT, nB, -1]);
        flogits, _ = this.inflater.inflate(flogits, pred_length);
        seq_emb, segs = this.inflater.inflate(out_emb, pred_length);
        segs=list(segs.cpu().detach().numpy());
        pids = flogits[:, :-1].argmax(dim=1).detach().cpu();
        seq_emb=seq_emb.detach().cpu();
        pids=torch.split(pids,segs);
        seq_emb=torch.split(seq_emb,segs);
        return seq_emb,pids

    def set_etc(this, args):
        this.maxT = args["maxT"];
        this.inflater=neko_inflater();

    def set_loggers(this, log_path, name, args):
        try:
            if (args["measure_rej"]==True):
                this.logger_dict = {"accr":neko_oswr_Attention_AR_counter("[" + name + "]" + "test_accr", False),
                                    }
            else:
                this.logger_dict = {
                    "accr": neko_os_Attention_AR_counter("[" + name + "]" + "test_accr", False),
                    "loss": Loss_counter("[" + name + "]" + "train_accr"),
                };
        except:
            this.logger_dict={
                "accr": neko_os_Attention_AR_counter("[" + name + "]" + "test_accr", False),
                "loss": Loss_counter("[" + name + "]" + "train_accr"),
            };
    def test_topk_impl(this, data_dict, modular_dict, logger_dict,k):

        data, label, proto, plabel, tdict = \
            data_dict["image"], data_dict["label"], data_dict["proto"], data_dict["plabel"], data_dict["tdict"];
        preds = modular_dict["preds"];
        seq = modular_dict["seq"];
        sampler = modular_dict["sampler"];

        data = data.cuda();
        features = modular_dict["feature_extractor"](data)
        A, pred_length = modular_dict["CAM"](features)
        pred_length = pred_length.argmax(dim=-1)
        # A0=A.detach().clone();
        out_emb = seq(features[-1], A, None);
        # lossess = []
        beams_ = [];
        probs = [];
        # terms = [];

        loss = 0;
        nT, nB = out_emb.shape[0], out_emb.shape[1];
        logits = preds[0](out_emb.reshape([nT * nB, -1]), proto, plabel).reshape([nT, nB, -1]);
        logits, _ = this.inflater.inflate(logits, pred_length);
        idmat,beams = sampler.model.decode_beam_char(logits, pred_length, proto, plabel, tdict);
            # terms.append(terms_);

        return  idmat,beams,A;

    # No debugging outputs.
    def perf_impl(this,data_dict, modular_dict,logger_dict):
        data,label,proto, plabel, tdict= \
        data_dict["image"],data_dict["label"],data_dict["proto"],data_dict["plabel"],data_dict["tdict"];
        preds=modular_dict["preds"];
        seq=modular_dict["seq"];
        sampler=modular_dict["sampler"];


        data=data.cuda();
        features = modular_dict["feature_extractor"](data)
        A,pred_length = modular_dict["CAM"](features)
        pred_length=pred_length.argmax(dim=-1)
        # A0=A.detach().clone();
        out_emb =seq(features[-1],A,None);
        # lossess = []
        beams_ = [];
        probs = [];
        # terms = [];

        loss = 0;
        for i in range(len(preds)):
            nT,nB=out_emb.shape[0],out_emb.shape[1];
            logits = preds[i](out_emb.reshape([nT*nB]+list(out_emb.shape[2:])), proto,plabel).reshape([nT,nB,-1]);
            logits,_ = this.inflater.inflate(logits, pred_length);
            choutput, prdt_prob = sampler.model.decode(logits, pred_length, proto, plabel, tdict);
            beams_.append(choutput);
            probs.append(prdt_prob);
            # loss_, terms_ = modular_dict["losses"][i](proto, preds, label_flatten);
            # loss = loss_ +  loss;
            beams_.append(choutput);
            # terms.append(terms_);
        beams=[];
        for i in range(features[-1].shape[0]):
            beam = [];
            for j in range(len(beams_)):
                beam.append(beams_[j][i]);
            beams.append(beam)
        # A=A.max(dim=2)[0];
        flabel=[];
        if(label is not None):
            for l in label:
                s="";
                for c in regex.findall(r'\X', l, regex.U) :
                    if(c not in tdict):
                        s+="⑨";
                    else:
                        s+=c;
                flabel.append(s);
            logger_dict["accr"].add_iter(beams_[0],pred_length, flabel)
        rdict={}
        return beams_[0], rdict, beams;

    def test_impl(this,data_dict, modular_dict,logger_dict):
        data,label,proto, plabel, tdict= \
        data_dict["image"],data_dict["label"],data_dict["proto"],data_dict["plabel"],data_dict["tdict"];
        preds=modular_dict["preds"];
        seq=modular_dict["seq"];
        sampler=modular_dict["sampler"];


        data=data.cuda();
        features = modular_dict["feature_extractor"](data)
        A,pred_length = modular_dict["CAM"](features)
        pred_length=pred_length.argmax(dim=-1)
        # A0=A.detach().clone();
        out_emb =seq(features[-1],A,None);
        # lossess = []
        beams_ = [];
        probs = [];
        # terms = [];

        loss = 0;
        for i in range(len(preds)):
            nT,nB=out_emb.shape[0],out_emb.shape[1];
            logits = preds[i](out_emb.reshape([nT*nB]+list(out_emb.shape[2:])), proto,plabel).reshape([nT,nB,-1]);
            logits,_ = this.inflater.inflate(logits, pred_length);
            choutput, prdt_prob = sampler.model.decode(logits, pred_length, proto, plabel, tdict);
            beams_.append(choutput);
            probs.append(prdt_prob);
            # loss_, terms_ = modular_dict["losses"][i](proto, preds, label_flatten);
            # loss = loss_ +  loss;
            beams_.append(choutput);
            # terms.append(terms_);
        beams=[];
        for i in range(features[-1].shape[0]):
            beam = [];
            for j in range(len(beams_)):
                beam.append(beams_[j][i]);
            beams.append(beam)
        # A=A.max(dim=2)[0];
        flabel=[];
        if(label is not None):
            for l in label:
                s="";
                for c in regex.findall(r'\X', l, regex.U) :
                    if(c not in tdict):
                        s+="⑨";
                    else:
                        s+=c;
                flabel.append(s);
            logger_dict["accr"].add_iter(beams_[0],pred_length, flabel)

        try:
            A_d = modular_dict["CAM"].model.forward_d(features)
            aimAD = dump_att_ims(data, None, [A,A_d], pred_length, label);
            seq_emb, pids = this.dump_pred_raw(out_emb, modular_dict, proto, pred_length);
            all_im = [{"att": im} for im in aimAD];
            all_feat = [{"feat_res": (f,l)} for f,l in zip(seq_emb,pids)];
            rdict={"xtra_ims": all_im,"xtra_pts":all_feat};
        except:
            rdict={}
        return beams_[0], rdict, beams;

        # A.detach().reshape(A.shape[0], A.shape[1], A.shape[2], A.shape[3]).sum(2)
    def vis_logits_impl(this,img,data_dict,modular_dict,at_time):

        _, label, proto, plabel, tdict = \
            data_dict["image"], data_dict["label"], data_dict["proto"], data_dict["plabel"], data_dict["tdict"];
        data=img
        preds=modular_dict["preds"];
        seq=modular_dict["seq"];
        sampler=modular_dict["sampler"];


        data=data.cuda();
        data=torch.nn.Parameter(data,requires_grad=True);

        features = modular_dict["feature_extractor"](data)
        A,pred_length = modular_dict["CAM"](features)
        pred_length=pred_length.argmax(dim=-1)
        # A0=A.detach().clone();
        out_emb =seq(features[-1],A,None);
        # lossess = []
        beams_ = [];
        probs = [];
        # terms = [];

        loss = 0;
        nT,nB=out_emb.shape[0],out_emb.shape[1];
        logits = preds[0](out_emb.reshape([nT*nB]+list(out_emb.shape[2:])), proto,plabel).reshape([nT,nB,-1]);
        logits, _ = this.inflater.inflate(logits, pred_length);

        if (len(logits) <= at_time):
            return None;
        return logits[at_time]

    def pretest_impl(this,modular_dict,metaargs,**kwargs):
        rot = kwargs["rot"];
        normproto, plabel, tdict = modular_dict["sampler"].model.dump_all(metaargs=metaargs,use_sp=False);
        if (not rot):
            proto = modular_dict["prototyper"](normproto,use_sp=False);
        else:
            proto = modular_dict["prototyper"](normproto, rot);
        return {"proto":proto,"plabel":plabel,"tdict":tdict};



class neko_HDOS2C_eval_routine_CFmk7P(neko_HDOS2C_eval_routine_CFmk7):
    def pretest_impl(this,modular_dict,metaargs,**kwargs):
        rot = kwargs["rot"];
        normproto, plabel, tdict = modular_dict["sampler"].model.dump_all(metaargs=metaargs,use_sp=False);
        if (not rot):
            proto,att = modular_dict["prototyper"].model.forward_debug(normproto,use_sp=False);
        else:
            proto,att = modular_dict["prototyper"].model.forward_debug(normproto);
        # proto_a = modular_dict["prototyper"].model.__call__(normproto, use_sp=False);

        return {"proto":proto,"plabel":plabel,"tdict":tdict,"nproto":normproto,"catt":att};

    def test_impl(this,data_dict, modular_dict,logger_dict):
        data,label,proto, plabel, tdict= \
        data_dict["image"],data_dict["label"],data_dict["proto"],data_dict["plabel"],data_dict["tdict"];
        preds=modular_dict["preds"];
        seq=modular_dict["seq"];
        sampler=modular_dict["sampler"];


        data=data.cuda();
        features = modular_dict["feature_extractor"](data)
        A,pred_length = modular_dict["CAM"](features)
        pred_length=pred_length.argmax(dim=-1)
        # A0=A.detach().clone();
        out_emb =seq(features[-1],A,None);
        # lossess = []
        beams_ = [];
        probs = [];
        # terms = [];

        loss = 0;
        nT,nB=out_emb.shape[0],out_emb.shape[1];
        # logits,[asel,ret] = preds[0].model.forward_d(out_emb.reshape([nT*nB]+list(out_emb.shape[2:])), proto,plabel);
        # ret=ret.reshape(nT,nB,-1);
        # ret,_ = this.inflater.inflate(ret, pred_length);

        logits = preds[0].model.forward(out_emb.reshape([nT * nB] + list(out_emb.shape[2:])), proto, plabel, );

        logits=logits.reshape([nT,nB,-1]);
        logits,_ = this.inflater.inflate(logits, pred_length);

        choutput, prdt_prob = sampler.model.decode(logits, pred_length, proto, plabel, tdict);
        beams_.append(choutput);
        probs.append(prdt_prob);
        # loss_, terms_ = modular_dict["losses"][i](proto, preds, label_flatten);
        # loss = loss_ + loss;
        beams_.append(choutput);
            # terms.append(terms_);
        beams=[];
        for i in range(features[-1].shape[0]):
            beam = [];
            for j in range(len(beams_)):
                beam.append(beams_[j][i]);
            beams.append(beam)
        # A=A.max(dim=2)[0];
        flabel=[];
        if(label is not None):
            for l in label:
                s="";
                for c in regex.findall(r'\X', l, regex.U) :
                    if(c not in tdict):
                        s+="⑨";
                    else:
                        s+=c;
                flabel.append(s);
            logger_dict["accr"].add_iter(beams_[0],pred_length, flabel)
        try:
            # comment me for perf
            seq_emb, pids = this.dump_pred_raw(out_emb, modular_dict, proto, pred_length);
            A_d = modular_dict["CAM"].model.forward_d(features)
            aimAD = dump_att_ims(data, None, [A,A_d], pred_length, label);
            all_im = [{"att": im} for im in aimAD];
            all_feat = [{"feat_res": (f,l)} for f,l in zip(seq_emb,pids)];
            rdict={"xtra_ims": all_im,"xtra_pts":all_feat};
        except:
            rdict={}
        return beams_[0], rdict, beams;

class neko_HDOS2C_eval_routine_CFmk7PT(neko_HDOS2C_eval_routine_CFmk7):

    def test_impl(this,data_dict, modular_dict,logger_dict):
        data,label,proto, plabel, tdict= \
        data_dict["image"],data_dict["label"],data_dict["proto"],data_dict["plabel"],data_dict["tdict"];
        preds=modular_dict["preds"];
        seq=modular_dict["seq"];
        sampler=modular_dict["sampler"];


        data=data.cuda();
        features_ffn,features_att = modular_dict["feature_extractor"](data)
        A,pred_length = modular_dict["CAM"]([f.contiguous() for f in features_att])
        pred_length=pred_length.argmax(dim=-1)
        # A0=A.detach().clone();
        out_emb =seq(features_ffn[-1].contiguous(),A,None);
        # lossess = []
        beams_ = [];
        probs = [];
        # terms = [];

        loss = 0;
        nT,nB=out_emb.shape[0],out_emb.shape[1];
        # logits,[asel,ret] = preds[0].model.forward_d(out_emb.reshape([nT*nB]+list(out_emb.shape[2:])), proto,plabel);
        # ret=ret.reshape(nT,nB,-1);
        # ret,_ = this.inflater.inflate(ret, pred_length);

        logits = preds[0].model.forward(out_emb.reshape([nT * nB] + list(out_emb.shape[2:])), proto, plabel, );

        logits=logits.reshape([nT,nB,-1]);
        logits,_ = this.inflater.inflate(logits, pred_length);

        choutput, prdt_prob = sampler.model.decode(logits, pred_length, proto, plabel, tdict);
        beams_.append(choutput);
        probs.append(prdt_prob);
        # loss_, terms_ = modular_dict["losses"][i](proto, preds, label_flatten);
        # loss = loss_ + loss;
        beams_.append(choutput);
            # terms.append(terms_);
        beams=[];
        for i in range(features_ffn[-1].shape[0]):
            beam = [];
            for j in range(len(beams_)):
                beam.append(beams_[j][i]);
            beams.append(beam)
        # A=A.max(dim=2)[0];
        flabel=[];
        if(label is not None):
            for l in label:
                s="";
                for c in regex.findall(r'\X', l, regex.U) :
                    if(c not in tdict):
                        s+="⑨";
                    else:
                        s+=c;
                flabel.append(s);
            logger_dict["accr"].add_iter(beams_[0],pred_length, flabel)
        try:
            # comment me for perf
            A_d = modular_dict["CAM"].model.forward_d(features_att)
            aimAD = dump_att_ims(data, None, [A,A_d], pred_length, label);
            seq_emb, pids = this.dump_pred_raw(out_emb, modular_dict, proto, pred_length);
            all_im = [{"att": im} for im in aimAD];
            all_feat = [{"feat_res": (f,l)} for f,l in zip(seq_emb,pids)];
            rdict={"xtra_ims": all_im,"xtra_pts":all_feat};
        except:
            rdict={}
        return beams_[0], rdict, beams;


class neko_HDOS2C_eval_routine_CFmk7m(neko_HDOS2C_eval_routine_CFmk7):
    def test_impl(this, data_dict, modular_dict, logger_dict):
        data, label, mask, proto, plabel, tdict = \
            data_dict["image"], data_dict["label"],data_dict["mask"], data_dict["proto"], data_dict["plabel"], data_dict["tdict"];
        preds = modular_dict["preds"];
        seq = modular_dict["seq"];
        sampler = modular_dict["sampler"];
        mask=mask.float();

        data = data.cuda();
        features = modular_dict["feature_extractor"](data)
        A, pred_length = modular_dict["CAM"](features)
        if(mask.shape[-1] != A.shape[-1]):
            mask=trnf.interpolate(mask,(A.shape[-2],A.shape[-1]));
        A=A*mask.cuda();
        pred_length = pred_length.argmax(dim=-1)
        # A0=A.detach().clone();
        out_emb = seq(features[-1], A, None);
        # lossess = []
        beams_ = [];
        probs = [];
        # terms = [];

        loss = 0;
        for i in range(len(preds)):
            nT, nB = out_emb.shape[0], out_emb.shape[1];
            logits = preds[i](out_emb.reshape([nT * nB, -1]), proto, plabel).reshape([nT, nB, -1]);
            logits, _ = this.inflater.inflate(logits, pred_length);
            choutput, prdt_prob = sampler.model.decode(logits, pred_length, proto, plabel, tdict);
            beams_.append(choutput);
            probs.append(prdt_prob);
            # loss_, terms_ = modular_dict["losses"][i](proto, preds, label_flatten);
            # loss = loss_ + loss;
            beams_.append(choutput);
            # terms.append(terms_);
        beams = [];
        for i in range(features[-1].shape[0]):
            beam = [];
            for j in range(len(beams_)):
                beam.append(beams_[j][i]);
            beams.append(beam)
        # A=A.max(dim=2)[0];
        logger_dict["accr"].add_iter(beams_[0], pred_length, label)
        try:
            aim = dump_att_ims(data, None, [A], pred_length, label);
            aim = [{"att": im} for im in aim];
            rdict = {"xtra_ims": aim};
        except:
            rdict = {}
        return beams_[0], rdict, beams;

class neko_HDOS2C_eval_routine_CFmk7m2(neko_HDOS2C_eval_routine_CFmk7):
    def test_impl(this, data_dict, modular_dict, logger_dict):
        data, label, mask, proto, plabel, tdict = \
            data_dict["image"], data_dict["label"],data_dict["mask"], data_dict["proto"], data_dict["plabel"], data_dict["tdict"];
        preds = modular_dict["preds"];
        seq = modular_dict["seq"];
        sampler = modular_dict["sampler"];
        mask=mask.float().cuda();

        data = data.cuda();
        features = modular_dict["feature_extractor"](data)
        A, pred_length = modular_dict["CAM"](features,mask)
        if(mask.shape[-1] != A.shape[-1]):
            mask=trnf.interpolate(mask,(A.shape[-2],A.shape[-1]));
        A=A*mask.cuda();
        pred_length = pred_length.argmax(dim=-1)
        # A0=A.detach().clone();
        out_emb = seq(features[-1], A, None);
        # lossess = []
        beams_ = [];
        probs = [];
        # terms = [];

        loss = 0;
        for i in range(len(preds)):
            nT, nB = out_emb.shape[0], out_emb.shape[1];
            logits = preds[i](out_emb.reshape([nT * nB, -1]), proto, plabel).reshape([nT, nB, -1]);
            logits, _ = this.inflater.inflate(logits, pred_length);
            choutput, prdt_prob = sampler.model.decode(logits, pred_length, proto, plabel, tdict);
            beams_.append(choutput);
            probs.append(prdt_prob);
            # loss_, terms_ = modular_dict["losses"][i](proto, preds, label_flatten);
            # loss = loss_ + loss;
            beams_.append(choutput);
            # terms.append(terms_);
        beams = [];
        for i in range(features[-1].shape[0]):
            beam = [];
            for j in range(len(beams_)):
                beam.append(beams_[j][i]);
            beams.append(beam)
        # A=A.max(dim=2)[0];
        logger_dict["accr"].add_iter(beams_[0], pred_length, label)
        try:
            aim = dump_att_ims(data, None, [A], pred_length, label);
            aim = [{"att": im} for im in aim];
            rdict = {"xtra_ims": aim};
        except:
            rdict = {}
        return beams_[0], rdict, beams;