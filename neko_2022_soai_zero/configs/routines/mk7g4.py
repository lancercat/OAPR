
class neko_cwag5_subroutine:
    def __init__(this,cnt=64,wprecon=1,withfrecon=True,detach_proto_mva=False,domx_shuf=False,domx_proto=True,domx_feat=True):
        this.CNT=cnt;
        this.wprecon=wprecon;
        this.withfrecon=withfrecon;
        this.mva_detach_proto=detach_proto_mva;
        this.domx_shuf=domx_shuf;
        this.domx_proto=domx_proto;
        this.domx_feat=domx_feat;
    def get_chunk(this,rec_list,mylength):
        if(len(rec_list)==0):
            return [0,mylength];
        else:
            return [rec_list[-1].shape[0],rec_list[-1].shape[0]+mylength];


    def fp_impl(this,fout_emb,proto_,normprotos_,plabels,label_flatten,device,modular_dict):
        loss=torch.tensor(0,device=device).float();
        psel=torch.randperm(proto_.shape[0])[:this.CNT].to(proto_.device);
        proto=proto_[psel];
        normprotos_cu=torch.cat(normprotos_).to(proto_.device);
        normprotos=normprotos_cu[psel];
        terms={};

        if(modular_dict["p_recon"]!="NEP_skipped_NEP"):
            rec_list = [];
            rec_label=[];
            rec_dict={};
            rec_trunks=[];

            if(modular_dict["p_recon_loss"]!="NEP_skipped_NEP"):
                rec_list.append(proto)
                rec_trunks.append(this.get_chunk(rec_trunks,proto.shape[0]));
                rec_dict["proto"]=len(rec_dict);
            if((modular_dict["recon_char_fe"]!="NEP_skipped_NEP")or (modular_dict["fpm_recon_loss"]!="NEP_skipped_NEP")):
                fsela = torch.randperm(fout_emb.shape[0]);
                flaba = label_flatten[fsela];
                fsel = fsela[flaba != plabels[-1]];
                fsel = fsel[:this.CNT].to(proto_.device)
                normed_fout = trnf.normalize(fout_emb[fsel], dim=-1, p=2);
                flab = label_flatten[fsel];
                rec_label.append(flab);
                rec_dict["feat"]=len(rec_list);
                rec_trunks.append(this.get_chunk(rec_list,normed_fout.shape[0]));
                rec_list.append(normed_fout);

            if (modular_dict["shuf_img"] != "NEP_skipped_NEP"):
                shufp, mapping = modular_dict["shuf_img"](normprotos);
                pshufp = modular_dict["shuf_proto"]([shufp], use_sp=False);
                rec_dict["shuf"] = len(rec_list);
                rec_trunks.append(this.get_chunk(rec_list,pshufp.shape[0]));
                rec_list.append(pshufp);
                rec_label.append(torch.range(0,len(shufp),device=shufp.device)+plabels[-1]);

            recons = modular_dict["p_recon"](torch.cat(rec_list).unsqueeze(-1).unsqueeze(-1));
            if (this.wprecon > 0 and modular_dict["p_recon_loss"]!="NEP_skipped_NEP"):
                l_precon = modular_dict["p_recon_loss"](recons[rec_trunks[rec_dict["proto"]][0]:rec_trunks[rec_dict["proto"]][1]], normprotos);
                loss += l_precon;
                terms["p_recon"] = l_precon.item();

            if (modular_dict["shuf_img"] != "NEP_skipped_NEP"):
                l_precon_shuf = modular_dict["shuf_recon_loss"](recons[rec_trunks[rec_dict["shuf"]][0]:rec_trunks[rec_dict["shuf"]][1]], shufp);
                loss += l_precon_shuf;
                terms["shuf_recon"] = l_precon_shuf.item();
                if (modular_dict["shuf_part_recon_loss"] != "NEP_skipped_NEP"):
                    l_shufrec = modular_dict["shuf_part_recon_loss"](shufp, pshufp);
                    loss += l_shufrec;
                    terms["shuf_part_rec"] = l_shufrec.item();
                # A soft link to p_recon can be used here.

            if (this.withfrecon and modular_dict["recon_char_fe"] != "NEP_skipped_NEP"):
                labels=torch.cat([plabels[psel].to(label_flatten.device),label_flatten[fsel]]);
                # hard wired
                pred_f_recon = modular_dict["recon_char_fe"](recons[:rec_list[1][1]]);
                f_recon_logit = modular_dict["recon_char_pred"](pred_f_recon, proto_, plabels);
                l_frecon, _ = modular_dict["f_recon_loss"](None, f_recon_logit, labels);
                loss += l_frecon;
                terms["cyc"] = l_frecon.item();
            if(modular_dict["fpm_recon_loss"]!="NEP_skipped_NEP"):
                f_recon = recons[rec_trunks[rec_dict["feat"]][0]:rec_trunks[rec_dict["feat"]][1]];
                l_fpm = modular_dict["fpm_recon_loss"](f_recon, normprotos_cu, plabels[:-1], flab, dbgkey=None).mean();
                terms["fpm_recon"]=l_fpm.item();
                loss+=l_fpm;
            else:
                pass;


        if(modular_dict["dom_mix"]!="NEP_skipped_NEP"):
            dta=[]; # domain targets
            if(this.domx_feat==True):
                dta.append(normed_fout);
            elif(this.domx_feat=="detached"):
                dta.append(normed_fout.detach());

            if(this.domx_proto==True):
                dta.append(proto);
            elif (this.domx_proto == "detached"):
                dta.append(proto.detach());

            if(this.domx_shuf==True):
                dta.append(pshufp);

            l_dommix=modular_dict["dom_mix"](dta);
            loss += l_dommix;
            terms["dom_mix"] = l_dommix.item();
        return loss,terms;

class neko_HDOS2C_routine_CFmk7g5_rec_cyc3ks_core(neko_HDOS2C_routine_CFmk7g3_rec_cyc_core):
    def arm_submodules(this):
        this.inflater = neko_inflater();
        this.water_mod=neko_cwag5_subroutine();
    def fp_impl(this, input_dict,exdict, modular_dict,logger_dict,device):
        clips=input_dict["image"];

        # Prototypes(sampled)
        # And this helps using SYNTH words in LSCT
        target=exdict["target"];
        length=exdict["length"];
        tdict=exdict["tdict"];
        normprotos=exdict["proto"];
        # semb=exdict["semb"];
        plabel=exdict["plabel"];

        prototyper=modular_dict["prototyper"]

        proto=prototyper(normprotos,use_sp=False);
        label_flatten, length = flatten_label(target,EOSlen=0,length=length);
        target, label_flatten,culength = target.to(device), label_flatten.to(device),length.long().to(device);
        out_emb,A,pred_length=this.fe_seq(clips.to(device),modular_dict,length);
        fout_emb,_=this.inflater.inflate(out_emb,length)

        loss,terms,beams=recog_loss(modular_dict,pred_length,culength,fout_emb,proto,plabel,label_flatten,length,tdict);

        # computing accuracy and loss
        tarswunk = ["".join([tdict[i.item()] for i in target[j]]).replace('[s]', "") for j in
                    range(len(target))];
        logger_dict["accr"].add_iter(beams[0], length, tarswunk)
        logger_dict["loss"].add_iter(loss, terms[0])
        return loss;
