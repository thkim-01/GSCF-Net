import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# 필요한 모델 아키텍처 import
from transformers import RobertaConfig, RobertaModel, AutoModel # HuggingFace Hub 연동을 위해 AutoModel 추천
from models.ginet_finetune import GINet # GINet 경로는 그대로 사용

# -------------------------------------------------------------------
# 1. CrossAttentionFusion 모듈 (리팩터링된 버전)
# -------------------------------------------------------------------
class CrossAttentionFusion(nn.Module):
    def __init__(self, graph_dim: int, sequence_dim: int, hidden_dim: int, output_dim: int, 
                 alpha: float = 1.0, beta: float = 1.0):
        """
        CrossAttentionFusion 모듈 초기화
        
        Args:
            graph_dim (int): 그래프 임베딩의 차원.
            sequence_dim (int): 시퀀스 임베딩의 차원.
            hidden_dim (int): Q, K 벡터를 위한 내부 어텐션 차원.
            output_dim (int): 최종 출력 벡터의 차원.
            alpha (float): Key(K) 스케일링 상수. 어텐션 분포의 집중도를 조절.
            beta (float): Value(V) 스케일링 상수. 전달되는 정보의 총량을 조절.
        """
        super().__init__()
        
        # --- 상수 파라미터 저장 ---
        self.alpha = alpha
        self.beta = beta
        print(f"CrossAttentionFusion initialized with alpha={self.alpha}, beta={self.beta}")
        
        # --- 프로젝션 레이어 정의 (기존과 동일) ---
        self.q_graph = nn.Linear(graph_dim, hidden_dim)
        self.k_seq = nn.Linear(sequence_dim, hidden_dim)
        self.v_seq = nn.Linear(sequence_dim, graph_dim)

        self.q_seq = nn.Linear(sequence_dim, hidden_dim)
        self.k_graph = nn.Linear(graph_dim, hidden_dim)
        self.v_graph = nn.Linear(graph_dim, sequence_dim)

        self.norm_graph = nn.LayerNorm(graph_dim)
        self.norm_seq = nn.LayerNorm(sequence_dim)
        
        self.output_projection = nn.Linear(graph_dim + sequence_dim, output_dim) if output_dim is not None else None

    def forward(self, graph_features: torch.Tensor, sequence_features: torch.Tensor) -> torch.Tensor:
        graph_vecs = graph_features.unsqueeze(1)
        seq_vecs = sequence_features.unsqueeze(1)
        
        hidden_dim = self.q_graph.out_features 

        # --- 그래프 -> 시퀀스 어텐션 ---
        q_g = self.q_graph(graph_vecs)
        # k_seq와 v_seq에 alpha, beta 상수 적용
        k_s = self.k_seq(seq_vecs) * self.alpha
        v_s = self.v_seq(seq_vecs) * self.beta
        
        attn_scores_g = torch.bmm(q_g, k_s.transpose(1, 2)) / (hidden_dim ** 0.5) 
        attn_weights_g = F.softmax(attn_scores_g, dim=-1)
        seq_context = torch.bmm(attn_weights_g, v_s).squeeze(1)

        # --- 시퀀스 -> 그래프 어텐션 ---
        q_s = self.q_seq(seq_vecs)
        # k_graph와 v_graph에 alpha, beta 상수 적용
        k_g = self.k_graph(graph_vecs) * self.alpha
        v_g = self.v_graph(graph_vecs) * self.beta
        
        attn_scores_s = torch.bmm(q_s, k_g.transpose(1, 2)) / (hidden_dim ** 0.5) 
        attn_weights_s = F.softmax(attn_scores_s, dim=-1)
        graph_context = torch.bmm(attn_weights_s, v_g).squeeze(1)
        
        # --- 잔차 연결 및 정규화  ---
        fused_graph = self.norm_graph(graph_features + seq_context)
        fused_seq = self.norm_seq(sequence_features + graph_context)
        
        # --- 최종 융합 및 프로젝션  ---
        final_vec = torch.cat([fused_graph, fused_seq], dim=-1)
        if self.output_projection:
            final_vec = self.output_projection(final_vec)
        return final_vec

# -------------------------------------------------------------------
# 2. 전체 하이브리드 모델 정의 (config를 통해 alpha, beta 전달)
# -------------------------------------------------------------------
class HybridCrossAttentionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # --- 1. GNN 인코더 초기화 및 가중치 로드 (기존과 동일) ---
        print("Initializing GNN encoder...")
        self.gnn_encoder = GINet(**config['model'])
        # ... (가중치 로드 코드 생략, 기존과 동일) ...

        # --- 2. 언어 모델 인코더 로드 (기존과 동일) ---
        lm_model_path = config['cross_attention_specific']['chemberta_model_name']
        print(f"Initializing Language Model encoder from local path: {lm_model_path}")
        # `AutoModel` 사용을 권장하여 유연성 확보
        self.lm_encoder = AutoModel.from_pretrained(lm_model_path)
        print(f"Successfully loaded Language Model from local directory: {lm_model_path}")

        # --- 3. 교차-어텐션 융합 모듈 초기화 (config에서 alpha, beta 로드) ---
        print("Initializing Cross-Attention fusion layer...")
        fusion_config = config['cross_attention_specific']['fusion']
        # config 딕셔너리에 'graph_dim', 'sequence_dim' 추가
        fusion_config['graph_dim'] = config['model']['feat_dim']
        fusion_config['sequence_dim'] = self.lm_encoder.config.hidden_size
        fusion_config = config['cross_attention_specific']['fusion']
        # **fusion_config를 통해 alpha, beta를 포함한 모든 파라미터 전달
        # config 파일에 alpha, beta가 없으면 __init__의 기본값(1.0)이 사용됨
        self.fusion_layer = CrossAttentionFusion(**fusion_config)
        
        # --- 4. 최종 분류기 (MLP Head) 초기화 (기존과 동일) ---
        print("Initializing Classifier head...")
        clf_config = config['cross_attention_specific']['classifier']
        self.classifier = nn.Sequential(
            nn.Linear(fusion_config['output_dim'], clf_config['hidden_dim']),
            nn.ReLU(),
            nn.Dropout(clf_config['dropout']),
            nn.Linear(clf_config['hidden_dim'], config['dataset']['n_class'])
        )
        print("Model initialization complete.")

    def forward(self, graph_data, smiles_tokens):
        # (forward 로직은 기존과 완전히 동일)
        graph_features, _ = self.gnn_encoder(graph_data)
        lm_output = self.lm_encoder(**smiles_tokens)
        sequence_features = lm_output.pooler_output
        fused_features = self.fusion_layer(graph_features, sequence_features)
        logits = self.classifier(fused_features)
        return logits


# -------------------------------------------------------------------
# 3. 사용 예시 (리팩터링된 CrossAttentionFusion 테스트)
# -------------------------------------------------------------------
if __name__ == '__main__':
    # 가상 파라미터
    BATCH_SIZE = 16
    GRAPH_DIM = 300
    SEQ_DIM = 768
    HIDDEN_DIM = 128
    OUTPUT_DIM = 256
    
    # --- 실험을 위한 상수 파라미터 ---
    # 예: Key의 영향력을 20% 늘리고, Value의 정보량은 10% 줄여서 테스트
    ALPHA_CONST = 1.2 
    BETA_CONST = 0.9

    print("--- [Test 1] Default Parameters (alpha=1.0, beta=1.0) ---")
    # alpha, beta를 지정하지 않으면 기본값 1.0으로 동작
    default_attn_model = CrossAttentionFusion(
        graph_dim=GRAPH_DIM,
        sequence_dim=SEQ_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM
    )

    print("\n--- [Test 2] Custom Parameters ---")
    # 모델 생성 시 alpha, beta 상수 전달
    custom_attn_model = CrossAttentionFusion(
        graph_dim=GRAPH_DIM,
        sequence_dim=SEQ_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        alpha=ALPHA_CONST,
        beta=BETA_CONST
    )

    # 가상 입력 데이터
    dummy_graph_features = torch.randn(BATCH_SIZE, GRAPH_DIM)
    dummy_sequence_features = torch.randn(BATCH_SIZE, SEQ_DIM)

    # 모델 실행
    output_vector = custom_attn_model(dummy_graph_features, dummy_sequence_features)

    print("\n--- [Execution Result with Custom Parameters] ---")
    print(f"Input graph feature shape: {dummy_graph_features.shape}")
    print(f"Input sequence feature shape: {dummy_sequence_features.shape}")
    print(f"Output vector shape: {output_vector.shape}")
    assert output_vector.shape == (BATCH_SIZE, OUTPUT_DIM)
    print("\nRefactoring successful!")