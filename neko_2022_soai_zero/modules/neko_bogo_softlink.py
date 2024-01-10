# create a soft link of a module---
# while you can alternatively config whole model sharing with routine...
# But ``options are always welcome''

# Using this on bogo modules may yield problems, so please make hard copies
# they are links anyway, and a link to another link is kinda stupid.
class neko_bogo_softlink:
    def __init__(this,args,moddict):
        this.model=moddict[args["model"]];
    def __call__(this, *args, **kwargs):
        return this.model(*args,**kwargs);
    def cuda(this):
        pass;

