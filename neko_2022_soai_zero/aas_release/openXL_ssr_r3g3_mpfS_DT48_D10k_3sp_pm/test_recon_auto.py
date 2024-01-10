from configs import TAG
from eval_configs_recon_auto import dan_open_all
from neko_2022_soai_zero.do_experiments import do_experiments2


if __name__ == '__main__':
    #DEV="318prirC";
    DEV="MEOWS-ZeroDimension";

    DROOT="/home/lasercat/mount/project290//";
    MNAME=__file__.split("/")[-2];
    do_experiments2(dan_open_all,DROOT,MNAME,"_E0",DEV,tag=TAG,export_path="NEP_rawpath_NEP");


