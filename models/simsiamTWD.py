import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchvision.models import resnet50
import numpy as np
from scipy.fft import dct

class SEM(nn.Module):
    def __init__(self, L, V, tau, **kwargs):
        super().__init__()
        self.L = L
        self.V = V
        self.tau = tau

    def forward(self, x):
        logits = x.view(-1, self.L, self.V)
        taus = self.tau
        return F.softmax(logits / taus, -1).view(x.shape[0], -1)

def kl_twd_loss(features1, features2, pe, Bw, lam=0.1,temperature = 0.1,model='DCT'):

    if model == 'SM':
        features1 = F.softmax(features1,dim=1)
        features2 = F.softmax(features2,dim=1)
    elif model == 'SEM':
        sem = SEM(L=16, V=16, tau=temperature)
        features = sem(features)/16.0
    elif model == 'FullSPEM':
        features = F.softmax(features,dim=1)
    else:
        pe = F.normalize(pe,dim=1)
        features1 = F.normalize(features1,dim=1)
        features1 = torch.mm(features1,pe.T)/temperature
        features1 = F.softmax(features1, dim=1)

        features2 = F.normalize(features2,dim=1)
        features2 = torch.mm(features2,pe.T)/temperature
        features2 = F.softmax(features2, dim=1)

    # Cross entropy
    prob1 = torch.mm(features1, Bw)
    prob2 = torch.mm(features2, Bw)

    TWD = torch.sqrt((prob1 - prob2) ** 2 + 10e-12).sum(1)

    log_prob1 = torch.log(prob1 + 0.0001)
    log_prob2 = torch.log(prob2 + 0.0001)
    kl1 = ((log_prob1 - log_prob2) * prob1).sum(1)#torch.diag(cross_entropy1) - cross_entropy1
    kl2 = ((log_prob2 - log_prob1) * prob2).sum(1)#torch.diag(cross_entropy2) - cross_entropy2
    JD = kl1 + kl2

    loss = TWD + lam * JD

    return loss

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
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048): # bottleneck structure
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
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing. 
        The training is unstable and the loss oscillates.
        """

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x 

class SimSiamTWD(nn.Module):
    def __init__(self, simplicial_model='DCT', backbone=resnet50()):
        super().__init__()
        
        self.backbone = backbone
        self.projector = projection_MLP(backbone.output_dim)

        self.encoder = nn.Sequential( # f encoder
            self.backbone,
            self.projector
        )
        self.predictor = prediction_MLP()

        out_dim = 2048
        Bw = torch.eye(out_dim)
        self.Bw = torch.tensor(Bw, dtype=torch.float32).to('cuda:0')

        pe = dct(np.eye(out_dim), axis=0, norm='ortho')
        self.pe = torch.tensor(pe,dtype=torch.float32).to('cuda:0')

        self.temperature = 0.1
        self.simplicial_model = simplicial_model

    def forward(self, x1, x2,stopgrad=True):

        f, h = self.encoder, self.predictor
        z1, z2 = f(x1), f(x2)
        p1, p2 = h(z1), h(z2)

        if stopgrad:
            L1 = kl_twd_loss(p1, z2.detach(), self.pe, self.Bw, self.temperature,model=self.simplicial_model)
            L2 = kl_twd_loss(p2, z1.detach(), self.pe, self.Bw, self.temperature,model=self.simplicial_model)
        else:
            L1 = kl_twd_loss(z1, z2, self.pe, self.Bw, self.temperature,model=self.simplicial_model)
            L2 = L1
            #L2 = kl_twd_loss(p2, z1.detach(), self.pe, self.Bw, self.temperature,model=self.simplicial_model)

        L = (0.5*L1 + 0.5*L2).mean()

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












