import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
    
def NN_model_est_DSBI(L_train, Y_train, L_test, Y_test, batch_size=100, epochs=20, lr=1e-3):
    torch.set_default_dtype(torch.float64)
    batch_size = int(batch_size)
    epochs     = int(epochs)
    L_train = np.array(L_train, dtype=np.float64, copy=True)
    Y_train = np.array(Y_train, dtype=np.float64, copy=True)
    Y_test = np.array(Y_test, dtype=np.float64, copy=True)
    L_test  = np.array(L_test,  dtype=np.float64, copy=True)
    L_train_t = torch.from_numpy(L_train).permute(0, 2, 1).contiguous()
    L_test_t  = torch.from_numpy(L_test) .permute(0, 2, 1).contiguous()
    Y_train_t = torch.from_numpy(Y_train)
    Y_test_t = torch.from_numpy(Y_test)
    device = torch.device("cpu")
    L_train_t = L_train_t.to(device)
    L_test_t  = L_test_t.to(device)
    Y_train_t = Y_train_t.to(device)
    n_channels = L_train_t.shape[1]
    seq_len    = L_train_t.shape[2]
    n_out      = Y_train_t.shape[1]

    class Model(nn.Module):
        def __init__(self, in_channels, seq_len, out_dim):
            super(Model, self).__init__()
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels, 64, kernel_size=7, padding=0, dtype=torch.float64),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=5),
                nn.Conv1d(64, 64, kernel_size=7, padding=0, dtype=torch.float64),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=5),
                nn.Conv1d(64, 64, kernel_size=7, padding=0, dtype=torch.float64),
                nn.ReLU()
            )
            L1 = seq_len - 6
            L2 = (L1 - 5) // 5 + 1
            L3 = L2 - 6
            L4 = (L3 - 5) // 5 + 1
            reduced_len = L4 - 6
            flat_feats  = 64 * reduced_len
            self.NN = nn.Sequential(
                nn.Linear(flat_feats, 64, dtype=torch.float64),
                nn.ReLU(),
                nn.Linear(64, 32, dtype=torch.float64),
                nn.ReLU(),
                nn.Linear(32, out_dim, dtype=torch.float64)
            )

        def forward(self, x):
            z = self.conv_layers(x)
            z = z.view(z.size(0), -1)
            out = self.NN(z)
            return out

    model = Model(in_channels=n_channels, seq_len=seq_len, out_dim=n_out).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    def run_epoch(XL, Y, train=True):
        N = XL.size(0)
        model.train(train) 
        total = 0
        for i in range(0, N, batch_size): 
            Lb = XL[i:i+batch_size]
            Yb = Y[i:i+batch_size]
            pred = model(Lb)
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
    validation_list = []
    for j in range(epochs):
        train = run_epoch(L_train_t, Y_train_t, train=True)
        train_list.append(train)
        validation = run_epoch(L_test_t, Y_test_t, train=False)
        validation_list.append(validation)
        print(f"epoch {j}  train_mse={train:.4f}  val_mse={validation:.4f}")

    model.eval()
    with torch.no_grad():
        preds = model(L_test_t).cpu().numpy()
    return preds, train_list, validation_list
