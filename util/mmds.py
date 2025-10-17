import torch


class MMDS:
    def __init__(self, initial_lambda: float = 0.0, momentum: float = 0.9, gamma: float = 1.0):
        """
        动态调整 HD loss 权重的 MMDS 模块
        """
        self.lambda_val = float(initial_lambda)
        self.momentum = float(momentum)
        self.gamma = float(gamma)
        self.m = 0.0
        self.L_prev = 0

    def update_lambda(self, L_val: float):
        """
        用当前损失 L_val 更新 lambda
        """
        L_val = float(L_val)
        delta_L = max(0.0, self.L_prev - L_val)
        alpha = self.gamma * delta_L
        self.m = self.momentum * self.m + (1.0 - self.momentum) * alpha
        self.lambda_val += self.m
        self.L_prev = L_val

    def get_lambda(self) -> float:
        return float(self.lambda_val)

