import re
import matplotlib.pyplot as plt

# Raw log text (truncated to save space in this snippet).
# In practice, you'd paste the full user‑supplied block here:
log_text = """
plain34 epoch 1/200  train 4.260/4.05%  val 4.239/3.96%
plain34 epoch 2/200  train 3.980/6.93%  val 3.973/7.35%
plain34 epoch 3/200  train 3.765/10.83%  val 3.645/12.33%
plain34 epoch 4/200  train 3.546/14.49%  val 3.628/14.18%
plain34 epoch 5/200  train 3.362/17.89%  val 3.830/13.78%
plain34 epoch 6/200  train 3.173/21.33%  val 3.207/21.80%
plain34 epoch 7/200  train 2.962/25.41%  val 3.152/22.78%
plain34 epoch 8/200  train 2.763/29.39%  val 3.140/23.21%
plain34 epoch 9/200  train 2.588/33.07%  val 3.583/18.35%
plain34 epoch 10/200  train 2.454/35.93%  val 3.666/20.22%
plain34 epoch 11/200  train 2.322/38.62%  val 2.638/33.82%
plain34 epoch 12/200  train 2.231/40.93%  val 3.232/26.85%
plain34 epoch 13/200  train 2.145/42.60%  val 2.955/30.20%
plain34 epoch 14/200  train 2.075/44.62%  val 2.912/31.82%
plain34 epoch 15/200  train 2.025/45.57%  val 2.924/31.70%
plain34 epoch 16/200  train 1.967/46.77%  val 2.542/37.51%
plain34 epoch 17/200  train 1.905/48.64%  val 2.478/38.31%
plain34 epoch 18/200  train 1.883/49.22%  val 2.357/40.31%
plain34 epoch 19/200  train 1.837/50.10%  val 2.456/38.62%
plain34 epoch 20/200  train 1.827/50.18%  val 2.602/36.99%
plain34 epoch 21/200  train 1.782/51.56%  val 2.959/33.78%
plain34 epoch 22/200  train 1.771/51.78%  val 2.788/33.43%
plain34 epoch 23/200  train 1.731/52.63%  val 2.554/37.61%
plain34 epoch 24/200  train 1.712/53.23%  val 3.013/32.70%
plain34 epoch 25/200  train 1.712/53.15%  val 2.764/36.72%
plain34 epoch 26/200  train 1.688/53.64%  val 2.546/39.21%
plain34 epoch 27/200  train 1.664/54.07%  val 2.213/43.62%
plain34 epoch 28/200  train 1.645/54.59%  val 3.122/30.91%
plain34 epoch 29/200  train 1.636/55.11%  val 2.666/36.79%
plain34 epoch 30/200  train 1.626/55.19%  val 3.065/31.28%
plain34 epoch 31/200  train 1.606/55.85%  val 2.852/32.85%
plain34 epoch 32/200  train 1.595/55.94%  val 2.046/46.62%
plain34 epoch 33/200  train 1.590/56.09%  val 3.021/32.09%
plain34 epoch 34/200  train 1.569/56.68%  val 2.333/42.52%
plain34 epoch 35/200  train 1.583/56.13%  val 2.081/47.10%
plain34 epoch 36/200  train 1.564/57.02%  val 2.229/44.37%
plain34 epoch 37/200  train 1.553/57.09%  val 2.631/38.76%
plain34 epoch 38/200  train 1.546/57.19%  val 2.264/43.75%
plain34 epoch 39/200  train 1.536/57.42%  val 2.882/34.35%
plain34 epoch 40/200  train 1.546/57.44%  val 2.469/40.67%
plain34 epoch 41/200  train 1.531/57.55%  val 2.227/43.42%
plain34 epoch 42/200  train 1.517/58.07%  val 2.782/36.44%
plain34 epoch 43/200  train 1.512/58.04%  val 2.630/38.81%
plain34 epoch 44/200  train 1.501/58.35%  val 2.218/44.31%
plain34 epoch 45/200  train 1.508/58.31%  val 2.778/37.23%
plain34 epoch 46/200  train 1.516/57.93%  val 3.408/32.41%
plain34 epoch 47/200  train 1.497/58.13%  val 2.023/48.35%
plain34 epoch 48/200  train 1.506/58.53%  val 2.707/37.48%
plain34 epoch 49/200  train 1.483/59.04%  val 3.639/30.43%
plain34 epoch 50/200  train 1.493/58.51%  val 2.085/46.75%
plain34 epoch 51/200  train 1.474/59.19%  val 2.780/35.38%
plain34 epoch 52/200  train 1.470/59.22%  val 2.490/39.39%
plain34 epoch 53/200  train 1.460/59.18%  val 3.423/28.81%
plain34 epoch 54/200  train 1.461/59.40%  val 2.779/35.36%
plain34 epoch 55/200  train 1.464/59.27%  val 1.936/50.07%
plain34 epoch 56/200  train 1.467/59.10%  val 2.687/37.93%
plain34 epoch 57/200  train 1.459/59.39%  val 3.056/35.18%
plain34 epoch 58/200  train 1.459/59.37%  val 2.728/36.45%
plain34 epoch 59/200  train 1.444/59.88%  val 2.056/46.65%
plain34 epoch 60/200  train 1.450/59.64%  val 3.322/32.48%
plain34 epoch 61/200  train 1.442/60.09%  val 2.151/47.10%
plain34 epoch 62/200  train 1.441/59.77%  val 2.484/41.32%
plain34 epoch 63/200  train 1.447/59.73%  val 2.404/41.03%
plain34 epoch 64/200  train 1.440/59.77%  val 2.023/47.82%
plain34 epoch 65/200  train 1.449/59.90%  val 3.549/29.02%
plain34 epoch 66/200  train 1.428/60.11%  val 2.500/40.35%
plain34 epoch 67/200  train 1.444/59.98%  val 2.776/38.66%
plain34 epoch 68/200  train 1.424/60.25%  val 2.357/42.14%
plain34 epoch 69/200  train 1.411/60.56%  val 2.702/37.39%
plain34 epoch 70/200  train 1.430/60.24%  val 2.918/33.68%
plain34 epoch 71/200  train 1.433/60.09%  val 2.293/43.01%
plain34 epoch 72/200  train 1.425/60.20%  val 2.355/44.96%
plain34 epoch 73/200  train 1.417/60.52%  val 2.241/44.30%
plain34 epoch 74/200  train 1.421/60.38%  val 3.058/33.69%
plain34 epoch 75/200  train 1.420/60.16%  val 3.063/34.84%
plain34 epoch 76/200  train 1.423/60.28%  val 2.631/37.99%
plain34 epoch 77/200  train 1.423/60.47%  val 2.081/47.70%
plain34 epoch 78/200  train 1.404/60.83%  val 2.254/44.31%
plain34 epoch 79/200  train 1.415/60.54%  val 2.330/43.37%
plain34 epoch 80/200  train 1.408/60.73%  val 3.097/33.32%
plain34 epoch 81/200  train 1.412/60.62%  val 2.179/45.55%
plain34 epoch 82/200  train 1.404/60.83%  val 2.481/40.23%
plain34 epoch 83/200  train 1.404/61.09%  val 2.777/36.54%
plain34 epoch 84/200  train 1.417/60.68%  val 2.555/37.47%
plain34 epoch 85/200  train 1.398/60.93%  val 3.265/32.72%
plain34 epoch 86/200  train 1.395/60.90%  val 2.293/43.77%
plain34 epoch 87/200  train 1.424/60.52%  val 2.335/44.05%
plain34 epoch 88/200  train 1.396/61.15%  val 2.419/42.73%
plain34 epoch 89/200  train 1.392/61.23%  val 2.614/40.48%
plain34 epoch 90/200  train 1.393/61.12%  val 2.724/36.04%
plain34 epoch 91/200  train 1.385/61.59%  val 2.357/42.08%
plain34 epoch 92/200  train 1.413/60.82%  val 2.631/38.72%
plain34 epoch 93/200  train 1.382/61.43%  val 2.427/41.11%
plain34 epoch 94/200  train 1.388/61.27%  val 2.639/37.31%
plain34 epoch 95/200  train 1.406/60.96%  val 2.757/39.29%
plain34 epoch 96/200  train 1.388/61.15%  val 2.865/36.35%
plain34 epoch 97/200  train 1.382/61.35%  val 2.410/42.20%
plain34 epoch 98/200  train 1.379/61.57%  val 2.290/44.25%
plain34 epoch 99/200  train 1.394/61.06%  val 2.343/42.43%
plain34 epoch 100/200  train 1.397/61.10%  val 2.320/43.14%
plain34 epoch 101/200  train 0.832/76.50%  val 1.172/67.36%
plain34 epoch 102/200  train 0.629/82.06%  val 1.166/67.91%
plain34 epoch 103/200  train 0.547/84.26%  val 1.148/69.20%
plain34 epoch 104/200  train 0.482/86.28%  val 1.159/68.95%
plain34 epoch 105/200  train 0.426/87.67%  val 1.190/68.68%
plain34 epoch 106/200  train 0.382/89.18%  val 1.175/68.99%
plain34 epoch 107/200  train 0.338/90.57%  val 1.215/68.54%
plain34 epoch 108/200  train 0.298/91.78%  val 1.248/68.28%
plain34 epoch 109/200  train 0.269/92.67%  val 1.303/67.62%
plain34 epoch 110/200  train 0.237/93.68%  val 1.262/68.34%
plain34 epoch 111/200  train 0.212/94.45%  val 1.306/68.03%
plain34 epoch 112/200  train 0.193/94.98%  val 1.333/67.36%
plain34 epoch 113/200  train 0.169/95.81%  val 1.294/68.34%
plain34 epoch 114/200  train 0.152/96.38%  val 1.368/67.34%
plain34 epoch 115/200  train 0.137/96.83%  val 1.348/67.76%
plain34 epoch 116/200  train 0.124/97.16%  val 1.348/68.02%
plain34 epoch 117/200  train 0.118/97.33%  val 1.367/67.80%
plain34 epoch 118/200  train 0.106/97.70%  val 1.417/67.14%
plain34 epoch 119/200  train 0.107/97.67%  val 1.451/66.80%
plain34 epoch 120/200  train 0.105/97.73%  val 1.430/66.84%
plain34 epoch 121/200  train 0.092/98.03%  val 1.465/66.84%
plain34 epoch 122/200  train 0.087/98.25%  val 1.426/67.48%
plain34 epoch 123/200  train 0.086/98.23%  val 1.416/67.67%
plain34 epoch 124/200  train 0.088/98.19%  val 1.420/67.03%
plain34 epoch 125/200  train 0.092/98.00%  val 1.446/66.82%
plain34 epoch 126/200  train 0.086/98.28%  val 1.477/66.65%
plain34 epoch 127/200  train 0.088/98.08%  val 1.485/66.65%
plain34 epoch 128/200  train 0.093/97.99%  val 1.477/66.15%
plain34 epoch 129/200  train 0.093/97.95%  val 1.540/65.92%
plain34 epoch 130/200  train 0.094/97.95%  val 1.640/63.55%
plain34 epoch 131/200  train 0.095/97.93%  val 1.548/65.15%
plain34 epoch 132/200  train 0.102/97.67%  val 1.491/65.57%
plain34 epoch 133/200  train 0.103/97.65%  val 1.513/65.39%
plain34 epoch 134/200  train 0.114/97.23%  val 1.530/64.95%
plain34 epoch 135/200  train 0.119/97.20%  val 1.550/64.72%
plain34 epoch 136/200  train 0.116/97.26%  val 1.551/64.59%
plain34 epoch 137/200  train 0.122/97.02%  val 1.602/63.87%
plain34 epoch 138/200  train 0.117/97.21%  val 1.630/63.65%
plain34 epoch 139/200  train 0.126/96.88%  val 1.637/63.88%
plain34 epoch 140/200  train 0.133/96.57%  val 1.739/62.26%
plain34 epoch 141/200  train 0.148/96.14%  val 1.720/62.54%
plain34 epoch 142/200  train 0.155/95.87%  val 1.674/62.73%
plain34 epoch 143/200  train 0.157/95.91%  val 1.833/59.44%
plain34 epoch 144/200  train 0.159/95.93%  val 1.846/59.75%
plain34 epoch 145/200  train 0.151/95.92%  val 1.690/62.78%
plain34 epoch 146/200  train 0.149/96.12%  val 1.779/61.19%
plain34 epoch 147/200  train 0.140/96.46%  val 1.678/62.16%
plain34 epoch 148/200  train 0.135/96.59%  val 1.688/62.56%
plain34 epoch 149/200  train 0.144/96.24%  val 1.880/60.14%
plain34 epoch 150/200  train 0.142/96.28%  val 1.961/57.61%
plain34 epoch 151/200  train 0.065/98.66%  val 1.359/68.64%
plain34 epoch 152/200  train 0.036/99.49%  val 1.338/69.30%
plain34 epoch 153/200  train 0.029/99.66%  val 1.329/69.19%
plain34 epoch 154/200  train 0.026/99.69%  val 1.324/69.30%
plain34 epoch 155/200  train 0.022/99.78%  val 1.316/69.73%
plain34 epoch 156/200  train 0.021/99.79%  val 1.314/69.47%
plain34 epoch 157/200  train 0.019/99.83%  val 1.307/69.81%
plain34 epoch 158/200  train 0.019/99.81%  val 1.310/69.77%
plain34 epoch 159/200  train 0.017/99.84%  val 1.311/69.42%
plain34 epoch 160/200  train 0.017/99.86%  val 1.308/69.66%
plain34 epoch 161/200  train 0.016/99.85%  val 1.304/69.50%
plain34 epoch 162/200  train 0.016/99.88%  val 1.307/69.53%
plain34 epoch 163/200  train 0.015/99.92%  val 1.296/69.79%
plain34 epoch 164/200  train 0.015/99.88%  val 1.302/69.69%
plain34 epoch 165/200  train 0.014/99.89%  val 1.300/69.61%
plain34 epoch 166/200  train 0.014/99.87%  val 1.295/69.74%
plain34 epoch 167/200  train 0.014/99.89%  val 1.296/69.77%
plain34 epoch 168/200  train 0.013/99.93%  val 1.289/69.65%
plain34 epoch 169/200  train 0.013/99.91%  val 1.287/69.57%
plain34 epoch 170/200  train 0.012/99.93%  val 1.285/69.81%
plain34 epoch 171/200  train 0.013/99.93%  val 1.283/69.68%
plain34 epoch 172/200  train 0.012/99.93%  val 1.285/69.81%
plain34 epoch 173/200  train 0.012/99.92%  val 1.289/69.63%
plain34 epoch 174/200  train 0.012/99.92%  val 1.285/69.78%
plain34 epoch 175/200  train 0.012/99.93%  val 1.280/69.90%
plain34 epoch 176/200  train 0.012/99.92%  val 1.282/69.82%
plain34 epoch 177/200  train 0.011/99.94%  val 1.280/69.79%
plain34 epoch 178/200  train 0.011/99.94%  val 1.275/69.96%
plain34 epoch 179/200  train 0.011/99.94%  val 1.276/69.84%
plain34 epoch 180/200  train 0.011/99.93%  val 1.274/69.77%
plain34 epoch 181/200  train 0.011/99.93%  val 1.270/69.89%
plain34 epoch 182/200  train 0.011/99.94%  val 1.266/69.82%
plain34 epoch 183/200  train 0.011/99.95%  val 1.268/69.89%
plain34 epoch 184/200  train 0.011/99.94%  val 1.261/70.18%
plain34 epoch 185/200  train 0.011/99.94%  val 1.270/70.13%
plain34 epoch 186/200  train 0.010/99.94%  val 1.264/70.19%
plain34 epoch 187/200  train 0.011/99.95%  val 1.268/70.08%
plain34 epoch 188/200  train 0.010/99.95%  val 1.265/70.12%
plain34 epoch 189/200  train 0.011/99.96%  val 1.261/70.09%
plain34 epoch 190/200  train 0.011/99.94%  val 1.262/69.97%
plain34 epoch 191/200  train 0.011/99.94%  val 1.255/70.22%
plain34 epoch 192/200  train 0.010/99.95%  val 1.255/70.20%
plain34 epoch 193/200  train 0.010/99.94%  val 1.257/70.04%
plain34 epoch 194/200  train 0.010/99.94%  val 1.254/69.99%
plain34 epoch 195/200  train 0.010/99.94%  val 1.255/70.32%
plain34 epoch 196/200  train 0.011/99.94%  val 1.253/70.18%
plain34 epoch 197/200  train 0.010/99.96%  val 1.249/70.05%
plain34 epoch 198/200  train 0.010/99.97%  val 1.247/70.21%
plain34 epoch 199/200  train 0.010/99.95%  val 1.249/70.09%
plain34 epoch 200/200  train 0.010/99.95%  val 1.247/70.03%
"""

# Extract all lines that begin with "resnet34"
pattern = re.compile(
    r"plain34 epoch\s+(\d+)/\d+\s+train\s+([\d.]+)/([\d.]+)%\s+val\s+([\d.]+)/([\d.]+)%",
    re.I,
)

metrics = {}
for m in pattern.finditer(log_text.replace('\n', ' ')):  # single‑line for robustness
    epoch = int(m.group(1))
    if epoch not in metrics:  # drop duplicates, keep first
        metrics[epoch] = {
            "train_loss": float(m.group(2)),
            "train_acc": float(m.group(3)),
            "val_loss": float(m.group(4)),
            "val_acc": float(m.group(5)),
        }

# Sort by epoch
epochs = sorted(metrics)
train_loss = [metrics[e]["train_loss"] for e in epochs]
val_loss = [metrics[e]["val_loss"] for e in epochs]
train_acc = [metrics[e]["train_acc"] for e in epochs]
val_acc = [metrics[e]["val_acc"] for e in epochs]

# Plot 1: Loss
plt.figure()
plt.plot(epochs, train_loss, label="Train Loss")
plt.plot(epochs, val_loss, label="Validation Loss", linestyle="--")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("PlainNet‑34 Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.show()
plt.savefig("/localdisk/ssrivas9/prismatic-vlms/viz/" + "plain34_loss_plot.png")

# Plot 2: Accuracy
plt.figure()
plt.plot(epochs, train_acc, label="Train Accuracy")
plt.plot(epochs, val_acc, label="Validation Accuracy", linestyle="--")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("PlainNet‑34 Training vs Validation Accuracy")
plt.legend()
plt.grid(True)
plt.show()
# Save the plots to files
plt.savefig("/localdisk/ssrivas9/prismatic-vlms/viz/" + "plain34_accuracy_plot.png")
