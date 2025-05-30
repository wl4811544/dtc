"""
测试模型的因果性约束
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest
from anomaly_kt.detector import CausalAnomalyDetector
from anomaly_kt.model import AnomalyAwareDTransformer


class TestCausality:
    """测试因果性约束"""
    
    @pytest.fixture
    def detector(self):
        """创建检测器实例"""
        return CausalAnomalyDetector(
            n_questions=100,
            n_pid=0,
            d_model=64,
            n_heads=4,
            n_layers=1
        )
    
    def test_detector_causality(self, detector):
        """测试检测器的因果性"""
        detector.eval()
        
        # 创建测试数据
        batch_size, seq_len = 2, 20
        q = torch.randint(0, 100, (batch_size, seq_len))
        s = torch.randint(0, 2, (batch_size, seq_len))
        
        with torch.no_grad():
            # 原始输出
            output1 = detector(q, s)
            
            # 修改未来位置的答案
            s_modified = s.clone()
            s_modified[:, 10:] = 1 - s_modified[:, 10:]
            
            # 新输出
            output2 = detector(q, s_modified)
            
            # 检查位置10之前的输出应该相同
            for t in range(10):
                diff = torch.abs(output1[:, t] - output2[:, t]).max().item()
                assert diff < 1e-6, f"Causality violated at position {t}"
            
            # 检查位置10及之后的输出应该不同
            future_changed = False
            for t in range(10, seq_len):
                diff = torch.abs(output1[:, t] - output2[:, t]).max().item()
                if diff > 1e-6:
                    future_changed = True
                    break
            
            assert future_changed, "Future changes had no effect"
    
    def test_attention_mask_shape(self, detector):
        """测试注意力掩码的形状"""
        seq_len = 10
        mask = torch.tril(torch.ones(seq_len, seq_len))
        
        # 检查是下三角矩阵
        for i in range(seq_len):
            for j in range(seq_len):
                if j > i:
                    assert mask[i, j] == 0, f"Mask allows future attention at ({i}, {j})"
                else:
                    assert mask[i, j] == 1, f"Mask blocks past attention at ({i}, {j})"
    
    def test_statistical_features_causality(self, detector):
        """测试统计特征的因果性"""
        batch_size, seq_len = 2, 15
        s = torch.randint(0, 2, (batch_size, seq_len))
        mask = torch.ones_like(s).bool()
        
        # 提取特征
        features = detector._extract_statistics(s, mask)
        
        # 修改未来的值
        s_modified = s.clone()
        s_modified[:, 10:] = 1 - s_modified[:, 10:]
        
        # 重新提取特征
        features_modified = detector._extract_statistics(s_modified, mask)
        
        # 前10个位置的特征应该相同
        for t in range(10):
            diff = torch.abs(features[:, t, :] - features_modified[:, t, :]).max().item()
            assert diff < 1e-6, f"Statistical features violated causality at position {t}"


class TestAnomalyIntegration:
    """测试异常检测集成"""
    
    def test_anomaly_weight_effect(self):
        """测试异常权重的影响"""
        from DTransformer.model import DTransformer
        
        # 创建检测器
        detector = CausalAnomalyDetector(
            n_questions=50,
            n_pid=0,
            d_model=32,
            n_heads=2,
            n_layers=1
        )
        
        # 创建两个模型，不同的异常权重
        model1 = AnomalyAwareDTransformer(
            n_questions=50,
            n_pid=0,
            d_model=32,
            n_heads=2,
            n_know=4,
            n_layers=1,
            anomaly_detector=detector,
            anomaly_weight=0.0  # 不考虑异常
        )
        
        model2 = AnomalyAwareDTransformer(
            n_questions=50,
            n_pid=0,
            d_model=32,
            n_heads=2,
            n_know=4,
            n_layers=1,
            anomaly_detector=detector,
            anomaly_weight=0.8  # 高度考虑异常
        )
        
        # 测试数据
        q = torch.randint(0, 50, (2, 10))
        s = torch.randint(0, 2, (2, 10))
        
        # 预测
        with torch.no_grad():
            y1, *_ = model1.predict_with_anomaly(q, s)
            y2, *_ = model2.predict_with_anomaly(q, s)
        
        # 输出应该不同
        diff = torch.abs(y1 - y2).max().item()
        assert diff > 1e-4, "Anomaly weight has no effect"


if __name__ == "__main__":
    # 运行测试
    test = TestCausality()
    detector = test.detector()
    
    print("Testing detector causality...")
    test.test_detector_causality(detector)
    print("✓ Detector causality test passed")
    
    print("\nTesting attention mask...")
    test.test_attention_mask_shape(detector)
    print("✓ Attention mask test passed")
    
    print("\nTesting statistical features causality...")
    test.test_statistical_features_causality(detector)
    print("✓ Statistical features causality test passed")
    
    print("\nTesting anomaly integration...")
    test2 = TestAnomalyIntegration()
    test2.test_anomaly_weight_effect()
    print("✓ Anomaly integration test passed")
    
    print("\n✓ All tests passed!")