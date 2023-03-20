import numpy as np
from scipy import linalg 
import torch
from sklearn.utils import check_array
import torch.nn as nn
import torch.optim as optim
'''
1. 전체적으로 ESN을 학습에 있어서는 먼저 ESN에 알맞은 parameters를 입력하고 정의한다
2. train values를 esn모델에 learning을 통해 input w, W를 learning 하고 fix시킨다
3. train values의 learning된 weight들과 구하고자 하는 output을 linear regression에 fit 한다
3. test values를 esn모델에 learning 하지말고 통과 시킨다
4. test values의 learning된 weight들을 linear regression에 predict를 통해 구하고자 하는 target을 알 수 있다
'''
torch.autograd.set_detect_anomaly(True)
USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda:0' if USE_CUDA else 'cpu')
class ESN():
    def __init__(self, n_readout, 
                 resSize, damping=0.3, spectral_radius=None,
                 weight_scaling=1.25,initLen=0, random_state=42,inter_unit=torch.tanh, learning_rate=1e-2):
        
        self.resSize=resSize
        self.n_readout=n_readout # 마지막에 연결된 노드 갯수
        self.damping = damping  # 소실하는 정도로 모든 노드를 사용하지 않는다
        self.spectral_radius=spectral_radius # 1보다 작아야한다
        self.weight_scaling=weight_scaling
        self.initLen=initLen # 처음에 버릴 길이
        self.random_state=random_state
        self.inter_unit=inter_unit
        self.learning_rate = learning_rate
        self.Win=None # 학습하여 input weight가 있다면 넣어준다
        self.W=None # 학습하여 weight가 있다면 넣어준다
        torch.manual_seed(random_state) # torch에서 random값 고정
        self.out=None
        
        
    def init_fit(self,input):
        if input.ndim==1:
            input=input.reshape(1,-1)
        input = check_array(input, ensure_2d=True)
        n_feature, n_input = input.shape
        with torch.no_grad():
            W1 = torch.rand(self.resSize,self.resSize, dtype=torch.double,requires_grad=False).to(device) - 0.5
            self.Win1 = (torch.rand(self.resSize,1+n_feature, dtype=torch.double,requires_grad=False).to(device) - 0.5) * 1
            W2 = torch.rand(self.resSize,self.resSize, dtype=torch.double,requires_grad=False).to(device) - 0.5
            self.Win2 = (torch.rand(self.resSize,1+n_feature, dtype=torch.double,requires_grad=False).to(device) - 0.5) * 1
        print('Computing spectral radius...')
        #spectral_radius = max(abs(linalg.eig(W)[0]))  default
        print('done.')
         # 가중치 업데이트 과정 -> weight_scaling 값으로 나눈 값으로 가중치를 업데이트함. -> weight_scaling은 가중치 학습률이다.
        rhoW1 = max(abs(linalg.eig(W1)[0]))
        rhoW2 = max(abs(linalg.eig(W2)[0]))
        if self.spectral_radius == None:
            self.W1= W1*(self.weight_scaling/rhoW1)
            self.W2= W2*(self.weight_scaling/rhoW2)
        else:
            self.W1= W1*(self.weight_scaling/self.spectral_radius)
            self.W2= W2*(self.weight_scaling/self.spectral_radius)
        
        Yt=torch.DoubleTensor(input[:,self.initLen+1:]).to(device)
        
       
        x1 = torch.zeros((self.resSize,1)).type(torch.double).to(device)
        x2 = torch.zeros((self.resSize,1)).type(torch.double).to(device) # x의 크기는 n_레저버 * 1
      
        
       
        self.x1=x1
        self.out = input[:,n_input-1] #generative mode를 위한 input의 last value를 저장
       
        #### train the output by ridge regression
        # reg = 1e-8  # regularization coefficient
        #### direct equations from texts:
        # X_T = X.T
        # Wout = np.dot( np.dot(Yt,X_T), linalg.inv( np.dot(X,X_T) + \
        # reg*np.eye(1+inSize+resSize) ) )
        # using scipy.linalg.solve:
        '''
        reg = 1e-8
        out = linalg.solve(torch.matmul(self.X,self.X.T) + reg*torch.eye(1+n_feature+self.resSize), torch.matmul(X,Yt.T)).T
        
        out=np.array(out)
        out=torch.DoubleTensor(out).to(device)
        self.out=torch.DoubleTensor(out)
        print(self.out)
        '''
        WL= torch.rand(1,1+n_feature+self.resSize, dtype=torch.double,requires_grad=True).to(device)
        Wout= torch.rand(1,1+n_feature+self.resSize, dtype=torch.double,requires_grad=True).to(device)
        criterion = torch.nn.MSELoss()
        parameters=[WL,Wout]
        optimizer = optim.Adam([WL,Wout], self.learning_rate)
        latent = torch.zeros((self.n_readout,n_input-1)).type(torch.double).to(device)
        Y = torch.zeros((self.n_readout,n_input-1)).type(torch.double).to(device)
        
        for i in range(150):
            
            for t in range(n_input-1):
                u=torch.DoubleTensor(np.array(input[:,t],ndmin=2)).to(device)
                x1 = (1-self.damping)*x1 + self.damping*self.inter_unit( torch.matmul( self.Win1, torch.vstack([torch.DoubleTensor([1]),u]) ) + torch.matmul( self.W1, x1 ) )
                la= torch.matmul( WL, torch.vstack([torch.DoubleTensor([1]),u,x1]).detach()).to(device) 
            
                if t >= self.initLen:
                    latent[:,t-self.initLen] = la
            for j in range(n_input-1):
                m = torch.DoubleTensor(latent[:,j]).to(device)
                x2 = (1-self.damping)*x2+ self.damping*self.inter_unit( torch.matmul( self.Win2, torch.vstack([torch.DoubleTensor([1]),m]) ) + torch.matmul( self.W2, x2 ) )
                y = torch.matmul( Wout, torch.vstack([torch.DoubleTensor([1]),m,x2]).detach()).to(device) 
            
                if j >= self.initLen:
                    Y[:,j-self.initLen] = y
                
           
            loss = criterion(Y,Yt) # view를 하는 이유는 Batch 차원 제거를 위해
            optimizer.zero_grad()
            loss.backward(retain_graph=True) 
            optimizer.step() 
            print(i,loss.item())
        self.Wout = Wout
        return Y
        
        

    def fit(self,input): # 처음 학습을 시킬 때 사용 
        self=self.init_fit(input)
        return self # 계산된 weight들을 들고와서 regression에 사용한다
    
    def pre_fit(self,input): # 이미 학습을 시킨 후 w와 input w가 있을 때 사용
        if input.ndim==1:
            input=input.reshape(1,-1)
        input = check_array(input, ensure_2d=True)
        n_feature, n_input = input.shape
        
        if self.Win == None:    # 앞에서 학습을 안 시켰을 경우 아래 적용
            self.Win = (torch.rand(self.resSize,1+n_feature, dtype=torch.double,requires_grad=False).to(device) - 0.5) * 1
        if self.W == None:      # 앞에서 학습을 안 시켰을 경우 아래 적용
            if self.spectral_radius == None:
                W = torch.rand(self.resSize,self.resSize, dtype=torch.double,requires_grad=False).to(device) - 0.5
                rhoW = max(abs(linalg.eig(W)[0]))
                self.W= W*(self.weight_scaling/rhoW)
            else:
                self.W= W*(self.weight_scaling/self.spectral_radius)
        
        X = torch.zeros((1+n_feature+self.resSize,n_input-self.initLen-1)).type(torch.double)
        X=X.to(device)   # X의 크기는 n_레저버 * 1
        x = torch.zeros((self.resSize,1)).type(torch.double)    # x의 크기는 n_레저버 * 1
        x=x.to(device)
        
        for t in range(n_input):
            u=torch.DoubleTensor(np.array(input[:,t],ndmin=2)).to(device) # input에서 값을 하나씩 들고온다
            x = (1-self.damping)*x + self.damping*self.inter_unit(torch.matmul(self.Win, torch.vstack([torch.DoubleTensor([1]),u])) + torch.matmul( self.W, x ))
            if t >= self.initLen:
                X[:,t-self.initLen] = torch.vstack([torch.DoubleTensor([1]),u,x])[:,0]    
        return self.X # 계산된 weight들을 들고와서 regression에 사용한다
 
    def predict(self,outLen):    #gerative mode
        # run the trained ESN in a generative mode. no need to initialize here, 
        # because x is initialized with training data and we continue from there.
        x=self.x
       
        Y = torch.zeros((self.n_readout,outLen)).to(device)
        u = torch.DoubleTensor(self.out).to(device)
     
        for t in range(outLen):
            
            x = (1-self.damping)*x + self.damping*self.inter_unit( torch.matmul( self.Win, torch.vstack([torch.DoubleTensor([1]),u]) ) + torch.matmul( self.W, x ) )
            y = torch.matmul( self.Wout, torch.vstack([torch.DoubleTensor([1]),u,x])).to(device) 
            Y[:,t] = y
            # generative mode:
            u = y
        self.Y=Y
        return Y