import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)


def ana():
    data = pd.read_csv('../reports/llm_mix_simgle.csv')
    data = data[data['material_id'].apply(lambda x: 's' not in x)]
    print(len(data))
    y_true = data['result']
    y_pred = data['ypred']

    acc = accuracy_score(y_true, y_pred)
    p = precision_score(y_true, y_pred, average='binary')  # 二分类
    r = recall_score(y_true, y_pred, average='binary')  # 二分类
    f1 = f1_score(y_true, y_pred, average='binary')  # 二分类

    # 打印结果
    metric_info = (f"\n====================================\n"
                   f"准确率 (Accuracy): {acc * 100:.2f}%\n"
                   f"精确率 (Precision): {p * 100:.2f}%\n"
                   f"召回率 (Recall): {r * 100:.2f}%\n"
                   f"F1分数 (F1-score): {f1 * 100:.2f}%\n"
                   f"====================================\n")
    print(metric_info)

if __name__ == '__main__':
    ana()