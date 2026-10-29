import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

# ==================================================================
# GLOBAL SETTINGS
# ==================================================================
BATCH_SIZE = 64
TEST_BATCH_SIZE = 1000
EPOCHS = 10
SEED = 1000
LOGGING_INTERVAL = 10
RUNS_ROOT = Path("runs")

# ==================================================================
# DATA (load once)
# ==================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='./data', train=True, download=True, transform=transform),
    batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='./data', train=False, download=True, transform=transform),
    batch_size=TEST_BATCH_SIZE, shuffle=False)


# ==================================================================
# MAXOUT HELPERS
# ==================================================================
class Maxout2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, k=2):
        super().__init__()
        self.linears = nn.ModuleList(
            [nn.Conv2d(in_ch, out_ch, kernel_size) for _ in range(k)])
    def forward(self, x):
        return torch.stack([l(x) for l in self.linears], dim=0).max(dim=0).values


class MaxoutLinear(nn.Module):
    def __init__(self, in_features, out_features, k=2):
        super().__init__()
        self.linears = nn.ModuleList(
            [nn.Linear(in_features, out_features) for _ in range(k)])
    def forward(self, x):
        return torch.stack([l(x) for l in self.linears], dim=0).max(dim=0).values


# ==================================================================
# NETWORK
# ==================================================================
class Net(nn.Module):
    def __init__(self, act='relu'):
        super(Net, self).__init__()
        self.act_name = act
        self.is_maxout = (act == 'maxout')

        if self.is_maxout:
            self.conv1 = Maxout2d(1, 32, 5, k=2)
            self.conv2 = Maxout2d(32, 64, 5, k=2)
            self.fc1   = MaxoutLinear(64 * 4 * 4, 1024, k=2)
            self.fc2   = nn.Linear(1024, 10)
        else:
            self.conv1 = nn.Conv2d(1, 32, 5)
            self.conv2 = nn.Conv2d(32, 64, 5)
            self.fc1   = nn.Linear(64 * 4 * 4, 1024)
            self.fc2   = nn.Linear(1024, 10)

        self.drop = nn.Dropout(0.5)
        self.activations = {}

    def act(self, x):
        if self.act_name == 'relu':       return F.relu(x)
        if self.act_name == 'tanh':       return torch.tanh(x)
        if self.act_name == 'sigmoid':    return torch.sigmoid(x)
        if self.act_name == 'leaky_relu': return F.leaky_relu(x, negative_slope=0.01)
        if self.act_name == 'maxout':     return x
        raise ValueError("Unsupported activation: " + self.act_name)

    def forward(self, x):
        x = self.conv1(x);         self.activations['conv1_pre']  = x.detach()
        x = self.act(x);           self.activations['conv1_act']  = x.detach()
        x = F.max_pool2d(x, 2);    self.activations['conv1_pool'] = x.detach()
        x = self.conv2(x);         self.activations['conv2_pre']  = x.detach()
        x = self.act(x);           self.activations['conv2_act']  = x.detach()
        x = F.max_pool2d(x, 2);    self.activations['conv2_pool'] = x.detach()
        x = x.view(x.size(0), -1); self.activations['flatten']    = x.detach()
        x = self.fc1(x);           self.activations['fc1_pre']    = x.detach()
        x = self.act(x);           self.activations['fc1_act']    = x.detach()
        x = self.drop(x);          self.activations['drop']       = x.detach()
        x = self.fc2(x);           self.activations['fc2_pre']    = x.detach()
        x = F.softmax(x, dim=1);   self.activations['softmax']    = x.detach()
        return x


# ==================================================================
# INITIALIZERS
# ==================================================================
def init_xavier(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None: nn.init.zeros_(m.bias)

def init_kaiming(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None: nn.init.zeros_(m.bias)

def apply_init(model, init_name):
    if init_name == "xavier":   model.apply(init_xavier)
    elif init_name == "kaiming": model.apply(init_kaiming)
    return model


# ==================================================================
# SINGLE RUN
# ==================================================================
def run_experiment(run_name, activation, init_name, optimizer_name, lr,
                   epochs=EPOCHS):
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    log_dir = RUNS_ROOT / run_name
    writer = SummaryWriter(log_dir=str(log_dir))

    model = Net(act=activation).to(device)
    model = apply_init(model, init_name)

    if optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == "sgd_momentum":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_name == "adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=lr)
    else:
        raise ValueError("Unsupported optimizer: " + optimizer_name)

    print(f"\n{'='*70}")
    print(f"RUN: {run_name}")
    print(f"     act={activation} | init={init_name} | "
          f"opt={optimizer_name} | lr={lr}")
    print(f"{'='*70}")

    eps = 1e-13
    final_acc = 0.0
    final_loss = 0.0

    def train(epoch):
        model.train()
        criterion = nn.NLLLoss()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(torch.log(out + eps), target)
            loss.backward()
            optimizer.step()
            if batch_idx % LOGGING_INTERVAL == 0:
                n_iter = (epoch - 1) * len(train_loader) + batch_idx
                writer.add_scalar('train/loss', loss.item(), n_iter)

        # end-of-epoch logging
        n_iter = epoch * len(train_loader)
        for name, p in model.named_parameters():
            layer, attr = os.path.splitext(name)
            writer.add_histogram(f'{layer}/{attr[1:]}',
                                 p.detach().cpu().numpy(), n_iter)
        for name, a in model.activations.items():
            a = a.detach().cpu()
            writer.add_histogram(f'activations/{name}', a.numpy(), n_iter)
            writer.add_scalar(f'activations/{name}/mean', a.mean().item(), n_iter)
            writer.add_scalar(f'activations/{name}/std',  a.std().item(),  n_iter)
            writer.add_scalar(f'activations/{name}/min',  a.min().item(),  n_iter)
            writer.add_scalar(f'activations/{name}/max',  a.max().item(),  n_iter)

    def test(epoch):
        nonlocal final_acc, final_loss
        model.eval()
        test_loss, correct = 0, 0
        criterion = nn.NLLLoss(reduction='sum')
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                out = model(data)
                test_loss += criterion(torch.log(out + eps), target).item()
                pred = out.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        acc = 100. * correct / len(test_loader.dataset)
        final_acc, final_loss = acc, test_loss
        print(f"  Epoch {epoch:2d} | test loss {test_loss:.4f} | "
              f"accuracy {acc:.2f}%")
        n_iter = epoch * len(train_loader)
        writer.add_scalar('test/loss', test_loss, n_iter)
        writer.add_scalar('test/accuracy', acc, n_iter)

    for epoch in range(1, epochs + 1):
        train(epoch)
        test(epoch)

    writer.close()
    return final_acc, final_loss


# ==================================================================
# ALL CONFIGURATIONS
# Format: (RUN_NAME, ACTIVATION, INIT, OPTIMIZER, LR)
# ==================================================================
CONFIGS = [
    # --- lr = 0.01 experiments ---
    ("baseline_relu_default_adam_lr01",  "relu",       "default", "adam",         0.01),
    ("act_tanh_xavier_adam_lr01",        "tanh",       "xavier",  "adam",         0.01),
    ("act_sigmoid_xavier_adam_lr01",     "sigmoid",    "xavier",  "adam",         0.01),
    ("act_leakyrelu_kaiming_adam_lr01",  "leaky_relu", "kaiming", "adam",         0.01),
    ("opt_relu_kaiming_sgd_lr01",        "relu",       "kaiming", "sgd",          0.01),
    ("opt_relu_kaiming_sgdmom_lr01",     "relu",       "kaiming", "sgd_momentum", 0.01),
    ("opt_relu_kaiming_adagrad_lr01",    "relu",       "kaiming", "adagrad",      0.01),
    # --- lr = 0.001 re-runs ---
    ("baseline_relu_default_adam_lr001", "relu",       "default", "adam",         0.001),
    ("act_tanh_xavier_adam_lr001",       "tanh",       "xavier",  "adam",         0.001),
    ("act_sigmoid_xavier_adam_lr001",    "sigmoid",    "xavier",  "adam",         0.001),
    ("act_leakyrelu_kaiming_adam_lr001", "leaky_relu", "kaiming", "adam",         0.001),
    ("act_maxout_kaiming_adam_lr001",    "maxout",     "kaiming", "adam",         0.001),
]


# ==================================================================
# RUN ALL
# ==================================================================
if __name__ == "__main__":
    results = []
    for run_name, act, init, opt, lr in CONFIGS:
        acc, loss = run_experiment(run_name, act, init, opt, lr)
        results.append((run_name, act, init, opt, lr, acc, loss))

    # ---------- Summary ----------
    print("\n\n" + "=" * 90)
    print(f"{'RUN':<36} {'ACT':<11} {'INIT':<9} "
          f"{'OPTIM':<14} {'LR':>6} {'ACC%':>7}")
    print("-" * 90)
    for run_name, act, init, opt, lr, acc, loss in results:
        print(f"{run_name:<36} {act:<11} {init:<9} "
              f"{opt:<14} {lr:>6} {acc:>6.2f}%")
    print("=" * 90)

    # ---------- Save as CSV ----------
    csv_path = Path("results_summary.csv")
    with open(csv_path, "w") as f:
        f.write("run_name,activation,init,optimizer,lr,final_test_accuracy,"
                "final_test_loss\n")
        for run_name, act, init, opt, lr, acc, loss in results:
            f.write(f"{run_name},{act},{init},{opt},{lr},{acc:.2f},{loss:.4f}\n")
    print(f"\nResults saved to {csv_path.resolve()}")