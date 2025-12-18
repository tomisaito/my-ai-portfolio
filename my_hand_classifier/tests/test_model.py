
import pytest
import torch
import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import create_model

def test_model_structure():
    # モデルが正しく作れるかテスト
    model = create_model(num_classes=3) # pretrained=Falseは省略可能か定義に合わせる
    assert model is not None
    
def test_forward_pass():
    # 偽の画像データを入れて、エラーが出ないかテスト
    model = create_model(num_classes=3)
    dummy_input = torch.randn(1, 3, 224, 224) # 1枚, RGB, 224x224
    output = model(dummy_input)
    
    # 出力が3クラス分あるか？
    assert output.shape == (1, 3)
    # 異常値（NaN）が出ていないか？
    assert not torch.isnan(output).any()
