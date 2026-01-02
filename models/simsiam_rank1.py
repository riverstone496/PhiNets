import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchvision.models import resnet50
import numpy as np



def D(p, z, version='simplified'): # negative cosine similarity
    if version == 'original':
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize 
        z = F.normalize(z, dim=1) # l2-normalize 
        return -(p*z).sum(dim=1).mean()

    elif version == 'simplified':# same thing, much faster. Scroll down, speed test in __main__
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception



class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048):
        super().__init__()
        ''' page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out- 
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d. 
        This MLP has 3 layers.
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(hidden_dim)
            #nn.utils.weight_norm(nn.Linear(hidden_dim, out_dim),name='weight')
        )
        self.num_layers = 3
    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x):
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        else:
            raise Exception
        return x 


class prediction_MLP(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=1): # bottleneck structure
        super().__init__()
        ''' page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied 
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers. 
        The dimension of h’s input and output (z and p) is d = 2048, 
        and h’s hidden layer’s dimension is 512, making h a 
        bottleneck structure (ablation in supplement). 
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.w = nn.Parameter(torch.Tensor(np.random.normal(size=(in_dim,out_dim))))

        #self.layer2 = nn.Linear(hidden_dim, out_dim)
        #self.layer2 = nn.utils.weight_norm(nn.Linear(hidden_dim, out_dim),name='weight')
        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing. 
        The training is unstable and the loss oscillates.
        """

    def forward(self, x):
        #x = self.layer1(x)
        x = torch.mm(x,F.normalize(self.w,0))
        return x 

class SimSiamRank1(nn.Module):
    def __init__(self, backbone=resnet50()):
        super().__init__()
        
        self.backbone = backbone
        self.projector = projection_MLP(backbone.output_dim)

        self.encoder = nn.Sequential( # f encoder
            self.backbone,
            self.projector
        )
        self.predictor = prediction_MLP()
    
    def forward(self, x1, x2):

        f, h = self.encoder, self.predictor
        z1, z2 = f(x1), f(x2)
        p1, p2 = h(z1), h(z2)

        
        m = nn.Sigmoid()

        sig_p1 = m(p1)
        sig_p2 = m(p2)

        lam = 0.01
        K1 = torch.cat((torch.abs(p1-10),torch.abs(p1+10)),1)
        P1 = torch.exp(-K1/lam)
        a = torch.ones((512,1)).to(K1.device)/512
        b = torch.ones((2,1)).to(K1.device)/2

        u = a
        v = b
        #Sinkhorn iteration
        
        for iter in range(3):
            u = a/(torch.mm(K1,v))
            v = b/(torch.mm(K1.T,u))

        P1 = torch.mm(u*K1,torch.diag(v.flatten()))
        P1 = P1*512
        
        K2 = torch.cat((torch.abs(p2-10),torch.abs(p2+10)),1)
        P2 = torch.exp(-K2/lam)

        u = a
        v = b
        #Sinkhorn iteration

        for iter in range(3):
            u = a/(torch.mm(K2,v))
            v = b/(torch.mm(K2.T,u))

        P2 = torch.mm(u*K2,torch.diag(v.flatten()))
        P2 = P2*512



        #L = (P1*K1).sum()

        #print(0)
        L = -((P1[:,0].flatten()*2-1)*(sig_p2.flatten()*2-1).detach()).sum()# - ((P1[:,0]*2-1).detach()*(P2[:,0]*2-1)).sum()
        #L = -(P1[:,0]*torch.log(P2[:,0].detach())).mean() -(P2[:,0]*torch.log(P1[:,0].detach())).mean()# + kl_loss(P2[:,0],P1[:,0].flatten().detach())
        #L = loss(P1[:,0],label_p2_p.flatten().detach()) / 2# + loss2(sig_p2,label_p1_p.detach()) / 2
        #L = L*n
        return {'loss': L}






if __name__ == "__main__":
    model = SimSiam()
    x1 = torch.randn((2, 3, 224, 224))
    x2 = torch.randn_like(x1)

    model.forward(x1, x2).backward()
    print("forward backwork check")

    z1 = torch.randn((200, 2560))
    z2 = torch.randn_like(z1)
    import time
    tic = time.time()
    print(D(z1, z2, version='original'))
    toc = time.time()
    print(toc - tic)
    tic = time.time()
    print(D(z1, z2, version='simplified'))
    toc = time.time()
    print(toc - tic)

# Output:
# tensor(-0.0010)
# 0.005159854888916016
# tensor(-0.0010)
# 0.0014872550964355469












