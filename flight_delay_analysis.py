import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

# =======================
# Sample Data Generation
# =======================
n_samples = 100000

# Generate flight delay target
y_true = np.random.choice(['On-Time', 'Delayed'], size=n_samples, p=[0.45, 0.55])
# Predicted values for models
logistic_pred = y_true.copy()
rf_pred = y_true.copy()
# Add small noise to simulate misclassification
logistic_pred[np.random.choice(n_samples, 172, replace=False)] = 'Delayed'
rf_pred[np.random.choice(n_samples, 7, replace=False)] = 'On-Time'

# Logistic regression coefficients (simulated)
logistic_coefs = pd.DataFrame({
    'feature': ['late_aircraft_delay','taxi_out','weather_delay','dep_hour','distance','day_of_week','cancelled','month','air_time'],
    'coef': [112, 50, 30, 10, -5, -3, -1, -0.5, -0.2]
})

# Random forest feature importance (simulated)
rf_importance = pd.DataFrame({
    'feature': ['taxi_out','late_aircraft_delay','weather_delay','dep_hour','air_time','distance','month','day_of_week','cancelled'],
    'importance': [0.864,0.12,0.011,0.003,0.001,0.001,0,0,0]
})

# =======================
# Machine Learning Charts
# =======================
fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# 1. Confusion Matrix - Logistic Regression
cm_log = confusion_matrix(y_true, logistic_pred, labels=['On-Time','Delayed'])
sns.heatmap(cm_log, annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
axes[0,0].set_title('Logistic Regression\nConfusion Matrix (Acc≈99.9%)')
axes[0,0].set_xlabel('Predicted')
axes[0,0].set_ylabel('Actual')

# 2. Confusion Matrix - Random Forest
cm_rf = confusion_matrix(y_true, rf_pred, labels=['On-Time','Delayed'])
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=axes[0,1])
axes[0,1].set_title('Random Forest\nConfusion Matrix (Acc≈100%)')
axes[0,1].set_xlabel('Predicted')
axes[0,1].set_ylabel('Actual')

# 3. ROC Curves
fpr, tpr, _ = roc_curve([1 if x=='Delayed' else 0 for x in y_true], [1 if x=='Delayed' else 0 for x in logistic_pred])
fpr_rf, tpr_rf, _ = roc_curve([1 if x=='Delayed' else 0 for x in y_true], [1 if x=='Delayed' else 0 for x in rf_pred])
axes[0,2].plot(fpr, tpr, label='Logistic Reg (AUC=1.00)')
axes[0,2].plot(fpr_rf, tpr_rf, label='Random Forest (AUC=1.00)')
axes[0,2].plot([0,1],[0,1],'--', color='grey')
axes[0,2].set_title('ROC Curves — Both Models')
axes[0,2].set_xlabel('False Positive Rate')
axes[0,2].set_ylabel('True Positive Rate')
axes[0,2].legend()

# 4. Random Forest Feature Importance
sns.barplot(x='importance', y='feature', data=rf_importance, ax=axes[1,0], palette='summer')
axes[1,0].set_title('Feature Importance (Random Forest)')
axes[1,0].set_xlabel('Importance Score')
axes[1,0].set_ylabel('')

# 5. Logistic Regression Coefficients
sns.barplot(x='coef', y='feature', data=logistic_coefs, ax=axes[1,1], palette='Reds')
axes[1,1].set_title('Logistic Regression Coefficients (+ve = More Delay Risk)')
axes[1,1].set_xlabel('Coefficient Value')
axes[1,1].set_ylabel('')

# 6. Model Performance Comparison
axes[1,2].bar(['Logistic Regression','Random Forest'], [0.999,1.0], color=['blue','green'])
axes[1,2].set_title('Model Performance Comparison')
axes[1,2].set_ylabel('Score')
axes[1,2].set_ylim(0,1.1)
for i, v in enumerate([0.999,1.0]):
    axes[1,2].text(i, v+0.02, f'{v:.3f}', ha='center')

plt.tight_layout()
plt.show()

# =======================
# Flight Delay Analysis Charts (Simulated)
# =======================
n_flights = 1000000
hours = np.arange(24)
avg_delay_hour = np.random.uniform(20,75,size=24)

airports = ['DVL','RDM','SMX','MOT','STC','VEL','ASE','ACV','PRC','SUN','BGM','TVC','GUC','JAC','MSO']
avg_delay_airport = np.random.uniform(40,62,size=15)

# 1. Average Departure Delay by Hour
plt.figure(figsize=(16,4))
plt.plot(hours, avg_delay_hour, color='red')
plt.fill_between(hours, avg_delay_hour, alpha=0.2)
plt.title('Average Departure Delay by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Avg Delay (minutes)')
plt.show()

# 2. Top 15 Airports by Avg Delay
plt.figure(figsize=(12,6))
sns.barplot(x=avg_delay_airport, y=airports, palette='Reds_r')
plt.title('Top 15 Airports by Average Flight Delay')
plt.xlabel('Average Delay (minutes)')
plt.ylabel('Airport Code')
plt.show()

# 3. Heatmap: Day of Week × Hour of Day
days = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
heatmap_data = np.random.uniform(20,120,(7,24))
plt.figure(figsize=(16,4))
sns.heatmap(heatmap_data, xticklabels=hours, yticklabels=days, cmap='YlOrRd')
plt.title('Heatmap: Day of Week × Hour of Day → Average Delay')
plt.xlabel('Hour of Day')
plt.ylabel('Day of Week')
plt.show()

# 4. Pie chart: Delay distribution by period
labels = ['Morning','Afternoon','Evening','Night','Late Night']
sizes = [16.5,18.2,21.4,17.9,25.9]
plt.figure(figsize=(6,6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
plt.title('Average Delay Distribution by Day Period')
plt.show()

# 5. Flight Status Breakdown
status_counts = [466473, 556351, 0]
status_labels = ['On-Time','Delayed','Cancelled']
plt.figure(figsize=(6,6))
sns.barplot(x=status_labels, y=status_counts, palette='Set2')
plt.title('Flight Status Breakdown')
plt.ylabel('Number of Flights')
plt.show()
