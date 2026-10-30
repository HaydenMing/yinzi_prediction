"""五个融合主体；沿用旧版融合和损失，编码器已移到model.py。非原论文完整复现。"""
import torch
from torch import nn
from torch.nn import functional as F

NAMES=['ucmib_pns','graph_gated','hierarchical','parallel','lcmf']


class Attention(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.attn=nn.MultiheadAttention(d,2,dropout=0.1,batch_first=True)
        self.norm=nn.LayerNorm(d)
        self.ff=nn.Sequential(nn.Linear(d,2*d),nn.GELU(),nn.Linear(2*d,d))
        self.norm2=nn.LayerNorm(d)

    def forward(self,q,kv=None):
        kv=q if kv is None else kv
        x=self.norm(q+self.attn(q,kv,kv,need_weights=False)[0])
        return self.norm2(x+self.ff(x))


def past_neighbors(x,k=3):
    n=x.size(1)
    sim=F.normalize(x.detach(),dim=-1)@F.normalize(x.detach(),dim=-1).transpose(1,2)
    past=torch.ones(n,n,dtype=torch.bool,device=x.device).tril(-1)
    sim=sim.masked_fill(~past,float('-inf'))
    indices=sim.topk(min(k,n),dim=-1).indices
    mask=torch.zeros_like(sim,dtype=torch.bool).scatter_(-1,indices,True)&past
    return mask|torch.eye(n,dtype=torch.bool,device=x.device)[None]


class GraphAttention(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.edge=nn.Sequential(nn.Linear(3*d,d),nn.LeakyReLU(.2),nn.Linear(d,1))
        self.value=nn.Linear(d,d)
        self.norm=nn.LayerNorm(d)

    def forward(self,x):
        n=x.size(1)
        a=x[:,:,None,:].expand(-1,-1,n,-1)
        b=x[:,None,:,:].expand(-1,n,-1,-1)
        logits=self.edge(torch.cat([a,b,(a-b).abs()],-1)).squeeze(-1)
        weights=logits.masked_fill(~past_neighbors(x),float('-inf')).softmax(-1)
        return self.norm(x+weights@self.value(x))


class Bottleneck(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.parameters_net=nn.Linear(d,2*d)

    def forward(self,x):
        mu,lv=self.parameters_net(x).chunk(2,-1)
        lv=lv.clamp(-5,3)
        z=mu+torch.exp(.5*lv)*torch.randn_like(mu) if self.training else mu
        return z,mu,lv


class FusionBackbone(nn.Module):
    def __init__(self,name,hidden=32):
        super().__init__()
        if name not in NAMES:
            raise ValueError(name)
        self.name=name; d=hidden
        self.hidden = hidden
        if name in ['ucmib_pns','parallel','lcmf']:
            self.pt=Attention(d); self.pf=Attention(d)
        if name in ['parallel','lcmf']:
            self.self_attention=Attention(d)
        if name=='hierarchical':
            self.tf=nn.Sequential(nn.Linear(2*d,d),nn.GELU())
            self.total=nn.Sequential(nn.Linear(2*d,d),nn.GELU())
        if name=='graph_gated':
            self.graphs=nn.ModuleList([GraphAttention(d) for _ in range(3)])
            self.gate=nn.Linear(3*d,3)
            self.base_gru=nn.GRU(d,d,batch_first=True)
            self.base_cnn=nn.Conv1d(d,d,3,padding=1)
            self.base_attn=Attention(d)
            self.base_heads=nn.ModuleList([nn.Linear(d,1) for _ in range(3)])
            self.mixer=nn.Sequential(nn.Linear(d,d//2),nn.GELU(),nn.Linear(d//2,3))
        elif name=='ucmib_pns':
            self.bottlenecks=nn.ModuleList([Bottleneck(d),Bottleneck(d)])
            self.branch_heads=nn.ModuleList([nn.Linear(d,1),nn.Linear(d,1)])
            self.full_head=nn.Linear(2*d,1)
            self.pns_estimator=nn.Sequential(nn.Linear(2*d,d),nn.GELU(),nn.Linear(d,4))
            self.head=nn.Sequential(nn.Linear(2*d,d//2),nn.GELU(),nn.Linear(d//2,1))
            self.register_buffer('queue_mu',torch.zeros(2,128,d))
            self.register_buffer('queue_lv',torch.zeros(2,128,d))
            self.register_buffer('queue_ids',torch.full((128,),-1,dtype=torch.long))
            self.register_buffer('queue_count',torch.tensor(0,dtype=torch.long))
            self.register_buffer('queue_pos',torch.tensor(0,dtype=torch.long))
        else:
            inp=d if name=='hierarchical' else (3*d if name=='parallel' else 2*d)
            self.head=nn.Sequential(nn.Linear(inp,d//2),nn.GELU(),nn.Dropout(.1),nn.Linear(d//2,1))

    def forward(self,xs):
        if len(xs) != 3 or any(x.ndim != 3 or x.size(-1) != self.hidden for x in xs):
            raise ValueError('融合主体需要三路[B,T,hidden]特征')
        p,t,f=xs
        if self.name=='hierarchical':
            h=self.total(torch.cat([p,self.tf(torch.cat([t,f],-1))],-1))
            pred=self.head(h[:,-1]).squeeze(-1)
        elif self.name=='parallel':
            h=torch.cat([self.self_attention(p),self.pt(p,t),self.pf(p,f)],-1)
            pred=self.head(h[:,-1]).squeeze(-1)
        elif self.name=='lcmf':
            enhanced=self.self_attention(p)
            h=torch.cat([self.pt(enhanced,t),self.pf(enhanced,f)],-1)
            pred=self.head(h[:,-1]).squeeze(-1)
        elif self.name=='graph_gated':
            views=[g(x) for g,x in zip(self.graphs,[p,t,f])]
            gates=self.gate(torch.cat(views,-1)).softmax(-1)
            h=sum(views[i]*gates[:,:,i:i+1] for i in range(3))
            reps=[self.base_gru(h)[0][:,-1],F.gelu(self.base_cnn(h.transpose(1,2))).mean(-1),self.base_attn(h)[:,-1]]
            base=torch.cat([head(rep) for head,rep in zip(self.base_heads,reps)],-1)
            weights=self.mixer(h[:,-1]).softmax(-1)
            return dict(prediction=(base*weights).sum(-1),base=base,fusion_features=h[:,-1])
        else:
            views=[self.pt(p,t)[:,-1],self.pf(p,f)[:,-1]]
            triples=[b(x) for b,x in zip(self.bottlenecks,views)]
            zs,mu,lv=[torch.stack([triple[i] for triple in triples],1) for i in range(3)]
            # 权重推理只依赖mu；真实y从未传入forward。
            probabilities=self.pns_estimator(mu.flatten(1)).sigmoid().reshape(-1,2,2)
            weights=probabilities.sum(-1)
            fused=(zs*weights[:,:,None]).flatten(1)
            pred=self.head(fused).squeeze(-1)
            branches=torch.cat([head(zs[:,i]) for i,head in enumerate(self.branch_heads)],-1)
            return dict(prediction=pred,fusion_features=fused,mu=mu,lv=lv,z=zs,probabilities=probabilities,
                        branches=branches,full=self.full_head(zs.flatten(1)).squeeze(-1))
        return dict(prediction=pred,fusion_features=h[:,-1])

    def loss(self,out,y,ids=None):
        loss=F.mse_loss(out['prediction'],y)
        if self.name!='ucmib_pns':
            return loss
        branch=out['branches']; full=out['full']; phi=.5
        # 训练期事实/反事实软标签；推理时不计算这些标签。
        with torch.no_grad():
            correct=(branch-y[:,None]).abs()<phi
            joint=(full-y).abs()<phi
            random_guess=torch.randn_like(y)
            absent_wrong=(random_guess-y).abs()>=phi
            necessary=joint[:,None]&~correct.flip(1)
            sufficient=correct&absent_wrong[:,None]
            targets=torch.stack([necessary,sufficient],-1).float()
        estimator=F.binary_cross_entropy(out['probabilities'],targets)
        aux=F.mse_loss(branch,y[:,None].expand_as(branch))+F.mse_loss(full,y)
        mu,lv=out['mu'],out['lv']; var=lv.exp()
        kl01=.5*(lv[:,1]-lv[:,0]+(var[:,0]+(mu[:,0]-mu[:,1]).square())/var[:,1]-1).mean()
        kl10=.5*(lv[:,0]-lv[:,1]+(var[:,1]+(mu[:,0]-mu[:,1]).square())/var[:,0]-1).mean()
        contrast=mu.new_tensor(0.)
        n=int(self.queue_count)
        for i in range(2):
            q=F.normalize(out['z'][:,i],dim=-1)
            key=F.normalize(mu[:,i]+torch.exp(.5*lv[:,i])*torch.randn_like(mu[:,i]),dim=-1)
            positive=(q*key).sum(-1,keepdim=True)/.2
            if n:
                # clone防止训练后入队修改反向图需要的缓存。
                qm=self.queue_mu[1-i,:n].detach().clone()
                ql=self.queue_lv[1-i,:n].detach().clone()
                negative=F.normalize(qm+torch.exp(.5*ql)*torch.randn_like(qm),dim=-1)
                logits=q@negative.T/.2
                if ids is not None:
                    logits=logits.masked_fill(ids[:,None]==self.queue_ids[:n][None],-1e4)
                logits=torch.cat([positive,logits],-1)
                contrast=contrast+F.cross_entropy(logits,torch.zeros(len(y),dtype=torch.long,device=y.device))
        return loss+.5*aux+.1*estimator+.01*contrast+.001*(kl01+kl10)

    @torch.no_grad()
    def enqueue(self,out,ids):
        if self.name!='ucmib_pns' or not self.training:
            return
        for j in range(len(ids)):
            pos=int(self.queue_pos)
            self.queue_mu[:,pos]=out['mu'][j].detach()
            self.queue_lv[:,pos]=out['lv'][j].detach()
            self.queue_ids[pos]=ids[j]
            self.queue_pos.fill_((pos+1)%128)
            self.queue_count.fill_(min(int(self.queue_count)+1,128))

