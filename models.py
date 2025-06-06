import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import matplotlib as mpl
from matplotlib.font_manager import FontProperties

# 设置中文字体
def set_chinese_font():
    try:
        # 尝试多个常见的中文字体
        font_list = ['SimHei', 'Microsoft YaHei', 'SimSun', 'FangSong', 'KaiTi']
        font = None
        for font_name in font_list:
            try:
                font = FontProperties(fname=f"C:\\Windows\\Fonts\\{font_name}.ttf")
                break
            except:
                continue
        return font
    except:
        return None

chinese_font = set_chinese_font()
# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BugPredictionModels:
    def __init__(self, processed_data_dir='processed_data'):
        self.data_dir = Path(processed_data_dir)
        self.projects = ['MozillaCore', 'Firefox', 'Thunderbird', 'EclipsePlatform', 'JDT']
        self.results_dir = Path('model_results')
        self.create_results_dir()

    def create_results_dir(self):
        """创建模型结果保存目录"""
        if not self.results_dir.exists():
            self.results_dir.mkdir(parents=True)
            logger.info(f"创建目录: {self.results_dir}")

    def load_processed_data(self, project_name):
        """加载预处理后的数据"""
        train_path = self.data_dir / f"{project_name}_train_processed.csv"
        test_path = self.data_dir / f"{project_name}_test_processed.csv"
        
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logger.info(f"成功加载 {project_name} 的处理后数据")
            return train_df, test_df
        except Exception as e:
            logger.error(f"加载数据时出错: {str(e)}")
            return None, None

    def prepare_features(self, df):
        """准备特征"""
        # 这里可以添加更多特征工程
        features = pd.DataFrame({
            'issue_id': df['Issue_id'],
            'is_duplicate': df['is_duplicate'],
            'duplicate_count': df['duplicate_count']
        })
        return features

    def evaluate_model(self, y_true, y_pred, y_prob=None):
        """评估模型性能"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred)
        }
        
        if y_prob is not None:
            metrics['auc'] = roc_auc_score(y_true, y_prob)
        
        return metrics

    def plot_confusion_matrix(self, y_true, y_pred, title, save_path):
        """绘制混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        if chinese_font:
            plt.title(title, fontproperties=chinese_font, fontsize=12)
            plt.ylabel('真实标签', fontproperties=chinese_font, fontsize=10)
            plt.xlabel('预测标签', fontproperties=chinese_font, fontsize=10)
        else:
            plt.title('Confusion Matrix', fontsize=12)
            plt.ylabel('True Label', fontsize=10)
            plt.xlabel('Predicted Label', fontsize=10)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_pr_curve(self, y_true, y_prob, title, save_path):
        """绘制PR曲线"""
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision)
        if chinese_font:
            plt.title(title, fontproperties=chinese_font, fontsize=12)
            plt.xlabel('召回率', fontproperties=chinese_font, fontsize=10)
            plt.ylabel('精确率', fontproperties=chinese_font, fontsize=10)
        else:
            plt.title('Precision-Recall Curve', fontsize=12)
            plt.xlabel('Recall', fontsize=10)
            plt.ylabel('Precision', fontsize=10)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def fault_prediction_model(self, project_name):
        logger.info(f"训练 {project_name} 的故障预测模型")
        train_df, test_df = self.load_processed_data(project_name)
        if train_df is None or test_df is None:
            return None
        X_train = self.prepare_features(train_df)
        X_test = self.prepare_features(test_df)
        y_train = train_df['is_duplicate']
        y_test = test_df['is_duplicate']
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train[['duplicate_count']], y_train)
        y_pred = model.predict(X_test[['duplicate_count']])
        y_prob = model.predict_proba(X_test[['duplicate_count']])[:, 1]
        metrics = self.evaluate_model(y_test, y_pred, y_prob)
        self.plot_confusion_matrix(
            y_test, y_pred,
            f"{project_name} 故障预测模型混淆矩阵",
            self.results_dir / f"{project_name}_fault_prediction_cm.png"
        )
        self.plot_pr_curve(
            y_test, y_prob,
            f"{project_name} 故障预测模型PR曲线",
            self.results_dir / f"{project_name}_fault_prediction_pr.png"
        )
        return metrics

    def anomaly_detection_model(self, project_name):
        """异常检测模型"""
        logger.info(f"训练 {project_name} 的异常检测模型")
        
        train_df, test_df = self.load_processed_data(project_name)
        if train_df is None or test_df is None:
            return None
        
        # 准备特征
        X_train = self.prepare_features(train_df)
        X_test = self.prepare_features(test_df)
        
        # 训练模型
        model = IsolationForest(contamination=0.1, random_state=42)
        model.fit(X_train[['duplicate_count']])
        # 预测 (-1为异常，1为正常，需要转换为0和1)
        y_pred = (model.predict(X_test[['duplicate_count']]) == -1).astype(int)
        y_true = test_df['is_duplicate']
        # 评估
        metrics = self.evaluate_model(y_true, y_pred)
        
        # 绘制评估图
        self.plot_confusion_matrix(
            y_true, y_pred,
            f"{project_name} 异常检测模型混淆矩阵",
            self.results_dir / f"{project_name}_anomaly_detection_cm.png"
        )
        return metrics

    def reusability_prediction_model(self, project_name):
        """重用性预测模型"""
        logger.info(f"训练 {project_name} 的重用性预测模型")
        
        train_df, test_df = self.load_processed_data(project_name)
        if train_df is None or test_df is None:
            return None
        
        # 准备特征和标签（将duplicate_count>1的标记为可重用）
        X_train = self.prepare_features(train_df)
        X_test = self.prepare_features(test_df)
        y_train = (train_df['duplicate_count'] > 1).astype(int)
        y_test = (test_df['duplicate_count'] > 1).astype(int)
        
        # 训练模型
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train[['duplicate_count']], y_train)
        
        # 预测
        y_pred = model.predict(X_test[['duplicate_count']])
        y_prob = model.predict_proba(X_test[['duplicate_count']])[:, 1]
        
        # 评估
        metrics = self.evaluate_model(y_test, y_pred, y_prob)
        
        # 绘制评估图
        self.plot_confusion_matrix(
            y_test, y_pred,
            f"{project_name} 重用性预测模型混淆矩阵",
            self.results_dir / f"{project_name}_reusability_prediction_cm.png"
        )
        
        self.plot_pr_curve(
            y_test, y_prob,
            f"{project_name} 重用性预测模型PR曲线",
            self.results_dir / f"{project_name}_reusability_prediction_pr.png"
        )
        
        return metrics

    def train_all_models(self):
        """训练所有项目的所有模型"""
        results = []
        
        for project in self.projects:
            project_results = {'Project': project}
            
            # 训练三个模型
            fault_metrics = self.fault_prediction_model(project)
            anomaly_metrics = self.anomaly_detection_model(project)
            reuse_metrics = self.reusability_prediction_model(project)
            
            if fault_metrics:
                project_results.update({
                    'Fault_' + k: v for k, v in fault_metrics.items()
                })
            
            if anomaly_metrics:
                project_results.update({
                    'Anomaly_' + k: v for k, v in anomaly_metrics.items()
                })
            
            if reuse_metrics:
                project_results.update({
                    'Reuse_' + k: v for k, v in reuse_metrics.items()
                })
            
            results.append(project_results)
        
        # 保存结果
        results_df = pd.DataFrame(results)
        results_df.to_csv(self.results_dir / 'model_results.csv', index=False)
        logger.info("所有模型训练完成，结果已保存")
        
        return results_df

if __name__ == "__main__":
    models = BugPredictionModels()
    results = models.train_all_models()
    print("\n模型评估结果:")
    print(results.to_string()) 