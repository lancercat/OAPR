import os.path
import pprint

def arm_hyperpara_280(srcdst):
    srcdst["val_each"]=10000;
    srcdst["vitr"]= 200000;
    srcdst["vepoch"]= 2;
    return srcdst;

def arm_hyperpara_fastdrop(srcdst):
    srcdst["val_each"]=10000;
    srcdst["vitr"]= 100000;
    srcdst["vepoch"]= 5;
    with open(os.path.join(srcdst["root"],"trainpara.log"),"w+") as fp:
        pprint.pprint(srcdst,fp);
    return srcdst;
