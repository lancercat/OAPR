import torch
from torch import nn

# Dan config.
from neko_sdk.encoders.chunked_resnet.bogo_nets import res45_net, res45_net_orig
from neko_sdk.encoders.chunked_resnet.layer_presets import res45_wo_bn, res45_bn, res45p_wo_bn, res45p_bn
from neko_sdk.encoders.chunked_resnet.neko_block_fe import norm_dict
# so this thing keeps the modules and
from osocrNG.bogomods_g2.res45g2_bws.res45_g2 import neko_res45_bogo_g2


class quick_layer_container:
    def collect(this,srcdst,container,namedict):
        for k in namedict:
            if(type(namedict[k])==dict):
                srcdst=this.collect(srcdst,container,namedict[k]);
            else:
                srcdst[namedict[k]]=container[namedict[k]];
        return srcdst;
    def __getitem__(this, item):
        return this.layers[item]
    def __init__(this,container,names):
        this.layers=this.collect({},container,names);
        this.name_dict=names;
class quick_modular:
    def __init__(this,any):
        this.model=any;
class neko_binorm_common(nn.Module):
    def __init__(this):
        super(neko_binorm_common, this).__init__()
    def freezebnprefix(this, prefix):
        for i in this.named_bn_dicts[prefix]:
            this.bns[i].eval();

    def unfreezebnprefix(this, prefix):
        for i in this.named_bn_dicts[prefix]:
            this.bns[i].train();

    def setup_bn_modules(this,mdict,prefix,gprefix):
        name_dict={};
        if(gprefix not in this.named_bn_dicts):
            this.named_bn_dicts[gprefix]=[];
        for k in mdict:
            if (type(mdict[k]) is dict):
                subdict=this.setup_bn_modules(mdict[k], prefix + "_" + k,gprefix);
                name_dict[k]=subdict;
            else:
                this.add_module(prefix + "_" + k, mdict[k]);
                this.named_bn_dicts[gprefix].append(len(this.bns));
                name_dict[k]=prefix + "_" + k;
                this.bns.append(mdict[k])
        return name_dict;
    def refresh_bogo(this):
        bogos = {}
        for i in range(len(this.bogo_names)):
            name = this.bogo_names[i];
            bn_name = this.bn_names[i];

            # bogos[name] = res45_net_orig(this.layer_names, this.named_bn_name_grps[bn_name],this._modules);
            bns=quick_layer_container(this._modules,this.named_bn_name_grps[bn_name]);
            layers=quick_layer_container(this._modules,this.layer_names);

            args={"mod_cvt":
            {
                "conv": "conv",
                "norm": "norm",
            },
            }
            cont={
                "conv":quick_modular(layers),
                "norm":quick_modular(bns),
            }
            bogos[name] = neko_res45_bogo_g2(args,cont);

        return bogos

    def freezebn(this):
        for i in this.bns:
            i.eval();
    def unfreezebn(this):
        for i in this.bns:
            i.train();

    def setup_modules(this,mdict,prefix):
        name_dict={};
        for k in mdict:
            if(type(mdict[k]) is dict):
                subname_dict=this.setup_modules(mdict[k],prefix+"_"+k);
                name_dict[k]=subname_dict;
            else:
                this.add_module(prefix+"_"+k,mdict[k]);
                name_dict[k]=prefix+"_"+k;
        return name_dict;

    def forward(self, input,debug=False):
        # This won't work as this is just a holder
        exit(9)
    def Iwantshapes(self):
        pseudo_input = torch.rand(1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
        features,grio = self.model(pseudo_input)
        return [feat.size()[1:] for feat in features]

class neko_r45_binorm_orig(neko_binorm_common):
    # inplace ReLUs cannot be trained in parallel.
    def __init__(this, strides, compress_layer, input_shape,bogo_names,bn_names,hardness=2,oupch=512,expf=1,ochs=None,inplace=False,bn_affine=True,drop=0):
        super(neko_r45_binorm_orig, this).__init__();
        if(strides is None):
            strides=[(1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1)];
        this.bogo_modules={};
        layers = res45_wo_bn(input_shape, oupch, strides,frac=expf,inplace=inplace);
        this.layer_names = this.setup_modules(layers,"shared_fe");

        this.bogo_names=bogo_names;
        this.bns=[];
        this.named_bn_dicts={};
        this.named_bn_name_grps={};
        this.bn_names=bn_names;
        for i in range(len(bogo_names)):
            bn_name = bn_names[i];
            bns = res45_bn( oupch, strides, frac=expf,affine=bn_affine);
            this.named_bn_name_grps[bn_name]=this.setup_bn_modules(bns, bn_name,bn_name);
        if (drop > 0):
            this.drop = torch.nn.Dropout(p=drop);
        else:
            this.drop = None;
        this.bogo_modules=this.refresh_bogo();

class neko_r45_binorm_origXN(neko_binorm_common):
    # inplace ReLUs cannot be trained in parallel.
    def __init__(this, strides, compress_layer, input_shape,bogo_names,norm_names,norm_types,hardness=2,oupch=512,expf=1,ochs=None,inplace=False,bn_affine=True,drop=0):
        super(neko_r45_binorm_origXN, this).__init__();
        if(strides is None):
            strides=[(1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1)];
        this.bogo_modules={};
        layers = res45_wo_bn(input_shape, oupch, strides,frac=expf,inplace=inplace);
        this.layer_names = this.setup_modules(layers,"shared_fe");

        this.bogo_names=bogo_names;
        this.bns=[];
        this.bn_names=norm_names;
        this.named_bn_dicts={};
        this.named_bn_name_grps={};
        this.norm_names=norm_names;
        for i in range(len(bogo_names)):
            bn_name = norm_names[i];
            bns = res45_bn(input_shape, oupch, strides, frac=expf,affine=bn_affine,engine=norm_dict()(norm_types[i]));
            this.named_bn_name_grps[bn_name]=this.setup_bn_modules(bns, bn_name,bn_name);
        if (drop > 0):
            this.drop = torch.nn.Dropout(p=drop);
        else:
            this.drop = None;
        this.bogo_modules=this.refresh_bogo();


class neko_r45_binorm_tpt(neko_binorm_common):
    def layeng(this):
        return res45_wo_bn,res45_bn;
    def __init__(this, strides, compress_layer, input_shape,bogo_names,bn_names,hardness=2,oupch=512,expf=1,inplace=True,drop=0):
        super(neko_r45_binorm_tpt, this).__init__()
        LAYER_ENG,BN_ENG=this.layeng();
        this.bogo_modules={};
        this.bogo_names=bogo_names;
        layers = LAYER_ENG(inpch=input_shape, oupch=oupch,strides= [(1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1)],frac=expf,inplace=inplace);
        this.layer_names = this.setup_modules(layers,"shared_fe");
        this.bns=[];
        this.bn_names=bn_names;
        this.tpt= neko_lens(int(32*expf),1,1,hardness);
        this.named_bn_dicts={};
        this.named_bn_name_grps={};
        for i in range(len(bogo_names)):
            bn_name = bn_names[i];
            bns = BN_ENG(input_shape, oupch, [(1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1)], frac=expf);
            this.named_bn_name_grps[bn_name]=this.setup_bn_modules(bns, bn_name,bn_name);
        if(drop>0):
            this.drop=torch.nn.Dropout(p=drop);
        else:
            this.drop=None;

        this.bogo_modules=this.refresh_bogo();


class neko_r45_binorm_ptpt(neko_r45_binorm_tpt):
    def layeng(this):
        return res45p_wo_bn,res45p_bn;


class neko_r45_binormXN_tpt(neko_binorm_common):
    def layeng(this):
        return res45_wo_bn,res45_bn;
    def __init__(this, strides, compress_layer, input_shape,bogo_names,norm_names,norm_types,hardness=2,oupch=512,expf=1,ochs=None,inplace=False,bn_affine=True,drop=0):
        super(neko_r45_binormXN_tpt, this).__init__()
        LAYER_ENG,BN_ENG=this.layeng();
        this.bogo_modules={};
        this.bogo_names=bogo_names;
        layers = LAYER_ENG(inpch=input_shape, oupch=oupch,strides= [(1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1)],frac=expf,inplace=inplace);
        this.layer_names = this.setup_modules(layers,"shared_fe");
        this.bns=[];
        this.bn_names=norm_names;
        this.tpt= neko_lens(int(32*expf),1,1,hardness);
        this.named_bn_dicts={};
        this.named_bn_name_grps={};
        for i in range(len(bogo_names)):
            bn_name = norm_names[i];
            bns = BN_ENG(input_shape, oupch, [(1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1)], frac=expf,affine=bn_affine,engine=norm_dict()(norm_types[i]));
            this.named_bn_name_grps[bn_name]=this.setup_bn_modules(bns, bn_name,bn_name);
        if(drop>0):
            this.drop=torch.nn.Dropout(p=drop);
        else:
            this.drop=None;

        this.bogo_modules=this.refresh_bogo();

class neko_r45_binormXN_ptpt(neko_r45_binormXN_tpt):
    def layeng(this):
        return res45p_wo_bn,res45p_bn;



class neko_r45_binorm_heavy_head(nn.Module):
    def setup_modules(this,mdict,prefix):
        for k in mdict:
            if(type(mdict[k]) is dict):
                this.setup_modules(mdict[k],prefix+"_"+k);
            else:
                this.add_module(prefix+"_"+k,mdict[k]);

    def __init__(this, strides, compress_layer, input_shape,bogo_names,bn_names,hardness=2,oupch=512,expf=1):
        super(neko_r45_binorm_heavy_head, this).__init__()
        this.bogo_modules={};
        ochs = [int(64*expf),int(64 * expf), int(64 * expf), int(128 * expf), int(256 * expf), oupch]

        layers = res45_wo_bn(input_shape, oupch, [(1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1)], 1,ochs=ochs);
        this.setup_modules(layers,"shared_fe");
        for i in range(len(bogo_names)):
            name = bogo_names[i];
            bn_name = bn_names[i];
            bns = res45_bn(input_shape, oupch, [(1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1)], 1);
            this.bogo_modules[name] = res45_net_orig(layers, bns);
            this.setup_modules(bns, bn_name);
       

    def forward(self, input,debug=False):
        # This won't work as this is just a holder
        exit(9)
    def Iwantshapes(self):
        pseudo_input = torch.rand(1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
        features,grio = self.model(pseudo_input)
        return [feat.size()[1:] for feat in features]



class neko_r45_binorm(neko_binorm_common):

    def __init__(this, strides, compress_layer, input_shape,bogo_names,bn_names,hardness=2,oupch=512,expf=1):
        super(neko_r45_binorm, this).__init__()
        this.bogo_modules={};
        this.bn_dict={};
        # grouped batch norm
        this.bngrps={};
        this.layers = res45_wo_bn(input_shape, oupch, [(1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1)], 1);
        this.setup_modules(layers, "shared_fe");
        this.bns=[];
        this.bogo_names=bogo_names;
        this.bn_names=bn_names;
        this.namedict={};
        for i in range(len(bogo_names)):
            bn_name=bn_names[i];
            bns = res45_bn(input_shape, oupch, [(1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1)], 1);
            this.setup_bn_modules(bns, bn_name);
        this.bogo_modules=this.refresh_bogo();


if __name__ == '__main__':
    layers=res45_wo_bn(3,512,[(1,1), (2,2), (1,1), (2,2), (1,1), (1,1)],1);
    bns=res45_bn(3,512,[(1,1), (2,2), (1,1), (2,2), (1,1), (1,1)],1);
    a=res45_net(layers,bns);
    t=torch.rand([1,3,32,128]);
    r=a(t);
    pass;
