import pandas as pd
import numpy as np
from pathlib import Path
import logging
import os

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BugDataPreprocessor:
    def __init__(self, base_path='.'):
        self.base_path = Path(base_path)
        self.projects = ['MozillaCore', 'Firefox', 'Thunderbird', 'EclipsePlatform', 'JDT']
        self.processed_dir = self.base_path / 'processed_data'
        
    def create_processed_dir(self):
        """创建处理后数据的存储目录"""
        if not self.processed_dir.exists():
            self.processed_dir.mkdir(parents=True)
            logger.info(f"创建目录: {self.processed_dir}")
    def load_data(self, project_name, data_type):
        file_path = self.base_path / project_name / f"{data_type}.csv"
        try:
            df = pd.read_csv(file_path)
            logger.info(f"成功加载 {project_name} 的 {data_type} 数据集")
            return df
        except Exception as e:
            logger.error(f"加载 {file_path} 时出错: {str(e)}")
            return None
    def process_duplicate_field(self, df):
        df['Duplicate'] = df['Duplicate'].replace('NULL', np.nan)
        df['is_duplicate'] = df['Duplicate'].notna().astype(int)
        df['duplicate_count'] = df['Duplicate'].fillna('').str.count(';') + df['is_duplicate']
        return df
    def save_processed_data(self, df, project_name, data_type):
        output_path = self.processed_dir / f"{project_name}_{data_type}_processed.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"已保存处理后的数据到: {output_path}")

    def process_project(self, project_name):
        """处理单个项目的数据
        
        Args:
            project_name (str): 项目名称
        """
        logger.info(f"开始处理项目: {project_name}")
        
        # 处理训练集
        train_df = self.load_data(project_name, 'train')
        if train_df is not None:
            train_df = self.process_duplicate_field(train_df)
            self.save_processed_data(train_df, project_name, 'train')
        
        # 处理测试集
        test_df = self.load_data(project_name, 'test')
        if test_df is not None:
            test_df = self.process_duplicate_field(test_df)
            self.save_processed_data(test_df, project_name, 'test')

    def process_all_projects(self):
        """处理所有项目的数据"""
        self.create_processed_dir()
        for project in self.projects:
            self.process_project(project)

    def generate_statistics(self):
        stats = []
        for project in self.projects:
            project_stats = {'Project': project}
            train_df = self.load_data(project, 'train')
            if train_df is not None:
                train_df = self.process_duplicate_field(train_df)
                project_stats.update({
                    'Train_Total': len(train_df),
                    'Train_Duplicates': train_df['is_duplicate'].sum(),
                    'Train_Duplicate_Rate': f"{(train_df['is_duplicate'].sum() / len(train_df)) * 100:.2f}%"
                })
            test_df = self.load_data(project, 'test')
            if test_df is not None:
                test_df = self.process_duplicate_field(test_df)
                project_stats.update({
                    'Test_Total': len(test_df),
                    'Test_Duplicates': test_df['is_duplicate'].sum(),
                    'Test_Duplicate_Rate': f"{(test_df['is_duplicate'].sum() / len(test_df)) * 100:.2f}%"
                })
            stats.append(project_stats)
        stats_df = pd.DataFrame(stats)
        stats_path = self.processed_dir / 'statistics.csv'
        stats_df.to_csv(stats_path, index=False)
        logger.info(f"已保存统计信息到: {stats_path}")
        return stats_df

if __name__ == "__main__":
    preprocessor = BugDataPreprocessor()
    preprocessor.process_all_projects()
    stats = preprocessor.generate_statistics()
    print("\n数据集统计信息:")
    print(stats.to_string()) 