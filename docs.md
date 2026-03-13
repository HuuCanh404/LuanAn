# tasks
- [ ] Define what “system resources” you want to predict
timestamp,
cpu%, ram%,
disk_read, disk_write,
net_in, net_out,
active_window_class,
window_count,
keyboard_rate,
mouse_rate,
process_count
- [ ] Define what “user behavior” means
- [ ] Collect data
- [x] Choose a baseline prediction method
- [x] Train and evaluate
- [ ] make a nix module


## things to skip (or do only when time allows)
GPU, process name



# Ba mo hinh AI (Three AI Models)

## Mo hinh 1: GradientBoostingRegressor (Regression)
- **Muc dich**: Du doan gia tri CPU % trong 5 giay toi
- **Thu vien**: `sklearn.ensemble.GradientBoostingRegressor`
- **Cau hinh**: 200 estimators, max_depth=6, learning_rate=0.1, subsample=0.8
- **Features**: Window 10 samples x 6 metrics (CPU, RAM, disk I/O, network) + statistics (mean, std, min, max, trend)
- **Uu diem**: Hieu qua tot voi du lieu tabular, chong overfitting bang subsample

## Mo hinh 2: RandomForestClassifier (Classification)
- **Muc dich**: Du doan xu huong CPU (Tang / On dinh / Giam)
- **Thu vien**: `sklearn.ensemble.RandomForestClassifier`
- **Cau hinh**: 200 estimators, max_depth=10, class_weight='balanced'
- **Nhan (Labels)**: 0=Giam (CPU giam >3%), 1=On dinh, 2=Tang (CPU tang >3%)
- **Uu diem**: Xu ly class imbalance bang balanced weights, dung cho du doan xu huong

## Mo hinh 3: MLPRegressor - Neural Network (Regression)
- **Muc dich**: Du doan gia tri CPU % bang neural network
- **Thu vien**: `sklearn.neural_network.MLPRegressor`
- **Cau hinh**: 3 hidden layers (256-128-64 neurons), ReLU activation, Adam optimizer, early stopping
- **Features**: Extended features voi moving averages, lag features, log transforms
- **Uu diem**: Co kha nang hoc cac moi quan he phi tuyen phuc tap


# prompt to give AI
Forecasting system resource usage based on user behavior.
the plans are 

Define what “system resources” is ( CPU usage %, RAM usage, Disk I/O, Network bandwidth, GPU usage,)
Define what “user behavior” means
Application usage (which apps are opened,Keyboard/mouse activity, Window focus changes, Command history, Website visits, Time-of-day patterns,)
Collect data
Choose a baseline prediction method ("CPU usage 5 seconds ahead" as base)
Train and evaluate
Iterate

# notes
event11
## qualms about different patterns between users:
Train and test on same user.
Train on 3 users, test on unseen 4th.

# What I'm currently doing
make a python script to log the system
nix home manager git
getting uv to work on nix

