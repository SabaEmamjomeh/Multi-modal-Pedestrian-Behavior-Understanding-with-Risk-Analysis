import pickle
import matplotlib.pyplot as plt
import os

# Load history dictionary
with open('./data_cache/train_stats.pkl', 'rb') as f:
    history = pickle.load(f)


epochs = history['epoch']

# Plot Losses
plt.figure(figsize=(6, 5))
plt.plot(epochs, history['loss'], label='Total Loss', linestyle='--', color='black')
plt.plot(epochs, history['intent_loss'], label='Intention Loss')
plt.plot(epochs, history['action_loss'], label='Action Loss')
plt.plot(epochs, history['risk_loss'], label='Risk Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('data_cache/loss_plot.png')
plt.close()

# Plot Accuracies and RMSE
plt.figure(figsize=(6, 5))
plt.plot(epochs, history['intent_acc'], label='Intention Accuracy')
plt.plot(epochs, history['action_acc'], label='Action Accuracy')
plt.plot(epochs, history['risk_rmse'], label='Risk RMSE')
plt.xlabel('Epoch')
plt.ylabel('Metric Value')
plt.title('Accuracy & RMSE over Epochs')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('data_cache/metrics_plot.png')
plt.close()

