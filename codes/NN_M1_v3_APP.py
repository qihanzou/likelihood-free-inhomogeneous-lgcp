import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
    
class Model(nn.Module):
    def __init__(self, n_channels, n_aux, n_out, seq_len):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=7, padding=0, dtype=torch.float64),
            nn.ReLU(),
            nn.MaxPool1d(5),
            nn.Conv1d(64, 64, kernel_size=7, padding=0, dtype=torch.float64),
            nn.ReLU(),
            nn.MaxPool1d(5),
            nn.Conv1d(64, 64, kernel_size=7, padding=0, dtype=torch.float64),
            nn.ReLU(),
        )

        L1 = seq_len - 6
        L2 = (L1 - 5) // 5 + 1
        L3 = L2 - 6
        L4 = (L3 - 5) // 5 + 1
        reduced_len = L4 - 6
        flat_feats  = 64 * reduced_len + n_aux  

        self.NN = nn.Sequential(
            nn.Linear(flat_feats, 64, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(64, 32, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(32, n_out, dtype=torch.float64),
        )
        self.n_aux = n_aux

    def forward(self, L_in, N_in):
        z = self.conv(L_in)          
        z = z.view(z.size(0), -1)    
        if self.n_aux > 0:
            if N_in.dim() == 1:
                N_in = N_in.unsqueeze(1)  
            z = torch.cat([z, N_in], dim=1)
        return self.NN(z)
      
def NN_model_est_M1(L_train, Y_train, L_test, N_train, N_test, batch_size=100, epochs=20, lr=1e-3):
    torch.set_default_dtype(torch.float64)
    device = torch.device("cpu")
    batch_size = int(batch_size)
    epochs = int(epochs)
    L_train = np.array(L_train, dtype=np.float64, copy=True, order="C")
    L_test  = np.array(L_test,  dtype=np.float64, copy=True, order="C")
    Y_train = np.array(Y_train, dtype=np.float64, copy=True, order="C")
    N_train = np.array(N_train, dtype=np.float64, copy=True, order="C")
    N_test  = np.array(N_test,  dtype=np.float64, copy=True, order="C")
    XL_train = torch.from_numpy(L_train).permute(0, 2, 1).contiguous().to(device)  
    XL_test  = torch.from_numpy(L_test ).permute(0, 2, 1).contiguous().to(device)  
    XN_train = torch.from_numpy(N_train).to(device)                                 
    XN_test  = torch.from_numpy(N_test ).to(device)                                 
    Y_train_t = torch.from_numpy(Y_train).to(device) 
    n_channels = XL_train.shape[1]
    n_aux      = XN_train.shape[1]
    n_out      = Y_train_t.shape[1]
    seq_len    = XL_train.shape[2]

    model   = Model(n_channels=n_channels, n_aux=n_aux, n_out=n_out, seq_len=seq_len).to(device)
    opt     = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
  
    def run_epoch(XL, XN, Y, train=True):
        N = XL.size(0)
        model.train(train) 
        total = 0
        for i in range(0, N, batch_size): 
            Lb = XL[i:i+batch_size]
            Nb = XN[i:i+batch_size]
            Yb = Y[i:i+batch_size]
            pred = model(Lb, Nb)
            loss = loss_fn(pred, Yb)
            if train:
               opt.zero_grad()
               loss.backward()
               opt.step()
            
            batch_size_i = Lb.size(0)
            Sum_of_losses_in_batch_i = loss.item()*batch_size_i
            total = total + Sum_of_losses_in_batch_i 
        return total / N

    train_list = []
    for j in range(epochs):
        train = run_epoch(XL_train, XN_train, Y_train_t, train=True)
        train_list.append(train)
        print(f"epoch {j}  train_mse={train:.4f}")

    model.eval()
    with torch.no_grad():
        preds = model(XL_test, XN_test).cpu().numpy()
    return preds, train_list
