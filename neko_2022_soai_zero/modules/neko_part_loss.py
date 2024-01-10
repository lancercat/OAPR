import torch
from torch import nn

from neko_2021_mjt.modulars.neko_inflater import neko_inflater;


class part_overlap_loss(nn.Module):
    def __init__(this,_):
        super(part_overlap_loss, this).__init__()
    def forward(this,A):
        B,T,P=A.shape[:3];
        SA=A.reshape(B,T,P,-1);
        n=max((P-1)*(P-1),0);
        overlap_loss=(SA.unsqueeze(-2)*SA.unsqueeze(-3)).sum(-1).triu(diagonal=1).sum(-1).sum(-1).mean()/n;
        return overlap_loss;


# prevents attention looking all over the place.
# We know that the model may want to model context, as it does help training.
# We also know that the language moves away, so, we prevent the model from doing so...
class att_distribute_loss(nn.Module):
    def __init__(this,args):
        super(att_distribute_loss, this).__init__();
        this.weight=args["weight"];
        this.inflator=neko_inflater();

    def make_mean(this,A,W,H):
        cord_grid = torch.stack(
            torch.meshgrid(torch.arange(0, H, device=A.device), torch.arange(0, W, device=A.device)), dim=-1);
        mean = (A.unsqueeze(-1) * cord_grid.unsqueeze(0)).sum(2).sum(2) / (
                A.unsqueeze(-1).sum(2).sum(2) + 0.0000009);
        return cord_grid,mean;
    def forward(this, A_, length):
        N, T, P, H, W = A_.shape;
        A = A_.permute(1, 0, 2, 3, 4);
        A, _ = this.inflator.inflate(A, length);
        cord_grid,mean=this.make_mean(A,W,H);
        dist = cord_grid.unsqueeze(0).unsqueeze(0) - mean.reshape(-1, P, 1, 1, 2);
        dist = (dist * dist).sum(-1);
        Asum=A.sum(-1).sum(-1);
        div_loss = (A * dist).sum(-1).sum(-1) / torch.max(Asum, torch.zeros_like(Asum)+0.000009);
        return div_loss.mean() / max(W, H)*this.weight;


# with detached cords
class att_distribute_loss_dtc(att_distribute_loss):
    def make_mean(this,A,W,H):
        with torch.no_grad():
            cord_grid = torch.stack(
                torch.meshgrid(torch.arange(0, H, device=A.device), torch.arange(0, W, device=A.device)), dim=-1);
            mean = (A.unsqueeze(-1) * cord_grid.unsqueeze(0)).sum(2).sum(2) / (
                    A.unsqueeze(-1).sum(2).sum(2) + 0.0000009);
        return cord_grid,mean;


class att_distribute_loss_dtcg3(att_distribute_loss_dtc):

    def forward(this, A_, length):
        N, T, P, H, W = A_.shape;
        A = A_.permute(1, 0, 2, 3, 4);
        A, _ = this.inflator.inflate(A, length);
        cord_grid,mean=this.make_mean(A,W,H);
        dist = cord_grid.unsqueeze(0).unsqueeze(0) - mean.reshape(-1, P, 1, 1, 2);
        dist = (dist * dist).sum(-1);
        Asum=A.sum(-1).sum(-1);
        div_loss = (A * dist).sum(-1).sum(-1) / torch.max(Asum, torch.zeros_like(Asum)+0.000009);
        return div_loss.mean() / min(W, H)*this.weight+\
            this.weight*torch.relu(0.8-A.reshape(A.shape[0],A.shape[1],-1).max(-1)[0]).sum(-1).mean();

