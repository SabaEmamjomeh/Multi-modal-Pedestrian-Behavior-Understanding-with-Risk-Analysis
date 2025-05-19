import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_squared_error


# Read the CSV file
df = pd.read_csv("./data_cache/scenario_results.csv")


def scenario_analysis(df, tag='scenario'):
    print(f"\n=== Scenario-based Metrics by {tag} ===")
    for value in df[tag].unique():
        sub = df[df[tag] == value]
        if len(sub) < 10:
            continue
        acc = accuracy_score(sub['intent_gt'], sub['intent_pred'])
        bacc = balanced_accuracy_score(sub['intent_gt'], sub['intent_pred'])
        traj_mse = sub['traj_mse'].mean()
        print(f"{value:40}: Acc={acc:.3f}, BAcc={bacc:.3f}, Traj MSE={traj_mse:.3f}")



scenario_analysis(df, tag='motion') 
scenario_analysis(df, tag='crossing') 
scenario_analysis(df, tag='age') 
scenario_analysis(df, tag='gender') 
scenario_analysis(df, tag='road_type') 

