import numpy as np
from tqdm import tqdm
import torch
from sklearn.model_selection import train_test_split
from get_circle_quarter import train


class LogisticRegressionSimple(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionSimple, self).__init__()        # 确保父类被正确初始化
        self.linear = torch.nn.Linear(input_dim, output_dim)    # 线性函数

    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))                 # sigmoid函数
        return outputs


def get_BiNormal():
    """生成双正态分布数据

    """
    
    np.random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)

    mu1 = -1.5 * torch.ones(2)  # 均值
    mu2 = 1.5 * torch.ones(2)   # 均值

    sigma1 = torch.eye(2) * 0.6 # 协方差矩阵
    sigma2 = torch.eye(2) * 1.2 # 协方差矩阵

    m1 = torch.distributions.MultivariateNormal(mu1, sigma1) # 多元正态分布
    m2 = torch.distributions.MultivariateNormal(mu2, sigma2) # 多元正态分布

    x1 = m1.sample((1000,)) # 生成1000个样本
    x2 = m2.sample((1000,)) # 生成1000个样本

    y1 = np.zeros(x1.size(0)) # 标签
    y2 = np.ones(x2.size(0))  # 标签

    X = torch.cat([x1, x2], dim=0)  # 拼接
    Y = np.concatenate([y1, y2])    # 拼接

    return X, Y



if __name__ == '__main__':
    # 生成双正态分布数据。X 为二维坐标，Y 为标签
    X, y = get_BiNormal() # shape: (2000, 2), (2000,)
    
    print(X.shape, y.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)

    input_dim = 2  # Two inputs x1 and x2
    output_dim = 1  # The output is 0/1 binary classification prob.
    model = LogisticRegressionSimple(input_dim, output_dim)
    train(X_train, X_test, y_train, y_test, learning_rate=0.01, model=model)
