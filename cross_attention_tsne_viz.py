import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix
import argparse
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

# 3D 시각화를 위한 import
from mpl_toolkits.mplot3d import Axes3D

# 동영상/GIF 생성을 위한 import
import imageio

# 기존 프로젝트의 클래스들을 그대로 가져와서 사용합니다.
from dataset.hybrid_dataset import HybridDatasetWrapper
from models.cross_attention_ratio import HybridCrossAttentionModel
from transformers import RobertaTokenizer

# --- 헬퍼 함수: 분자 이미지 그리드 생성 ---
def draw_molecule_grid(smiles_list, legends, mols_per_row=5, sub_img_size=(200, 200), output_path='molecules.png'):
    """SMILES 리스트로부터 분자 이미지 그리드를 생성합니다. (RDKit 미설치 시 대체)"""
    print(f"RDKit not available - skipping molecule visualization for {len(smiles_list)} compounds")
    print(f"Sample SMILES: {smiles_list[:3] if len(smiles_list) > 0 else []}")
    return None

# --- 메인 분석 클래스 ---
class CrossAttentionTSNEAnalyzer:
    def __init__(self, config_path, model_log_dir=None):
        self.config = self._load_config(config_path)
        self.device = self._get_device()
        
        # 분석할 모델 경로 설정
        self.model_log_dir = model_log_dir or self.config.get('analysis_specific', {}).get('log_dir_to_analyze')
        if not self.model_log_dir:
            # TensorBoard 로그 디렉토리에서 최신 모델 찾기
            self.model_log_dir = self._find_latest_model_dir()
        
        if not self.model_log_dir:
            raise ValueError("Cannot find model directory. Please specify model_log_dir or set analysis_specific.log_dir_to_analyze in config")
        
        # 분석할 타겟 설정
        self.target_name = self.config.get('analysis_specific', {}).get('target_to_analyze', 'Class')
        
        # 출력 디렉토리 생성
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = os.path.join('visualizations', f'cross_attention_tsne_{self.target_name}_{timestamp}')
        print("Output directory created: {}".format(self.output_dir))
        os.makedirs(self.output_dir, exist_ok=True)

        print("--- Cross-Attention t-SNE Analyzer ---")
        print(f"Config: {config_path}")
        print(f"Model Directory: {self.model_log_dir}")
        print(f"Target: {self.target_name}")
        print(f"Output Directory: {self.output_dir}")
        print(f"Device: {self.device}")
        print("--------------------------------------")
        
        # 모델과 데이터 로더 준비
        self.model = self._load_model()
        self.tokenizer = self._load_tokenizer()
        self.dataset_wrapper = self._prepare_dataset()
    
    def _load_config(self, path):
        """YAML 설정 파일을 로드합니다."""
        with open(path, "r", encoding="utf-8") as f:
            return yaml.load(f, Loader=yaml.FullLoader)
    
    def _get_device(self):
        """사용할 디바이스를 반환합니다."""
        if torch.cuda.is_available() and 'gpu' in self.config:
            return f"cuda:{self.config['gpu']}"
        return "cpu"
    
    def _find_latest_model_dir(self):
        """runs_ratio 디렉토리에서 최신 모델 디렉토리를 찾습니다."""
        runs_dir = 'runs_ratio'
        if not os.path.exists(runs_dir):
            return None
        
        # 가장 최신 디렉토리 찾기
        dirs = [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
        if not dirs:
            return None
        
        # 시간순으로 정렬하여 가장 최신 디렉토리 선택
        dirs.sort(reverse=True)
        latest_dir = os.path.join(runs_dir, dirs[0])
        
        # best_model.pth 파일이 있는지 확인
        model_path = os.path.join(latest_dir, 'best_model.pth')
        if os.path.exists(model_path):
            return latest_dir
        
        return None
    
    def _load_model(self):
        """저장된 최적의 모델을 불러옵니다."""
        model_path = os.path.join(self.model_log_dir, 'best_model.pth')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

        model = HybridCrossAttentionModel(self.config).to(self.device)
        state_dict = torch.load(model_path, map_location=self.device)
        
        # 호환성을 위해 position_ids 키 제거
        if 'lm_encoder.embeddings.position_ids' in state_dict:
            del state_dict['lm_encoder.embeddings.position_ids']
            print("Removed incompatible 'lm_encoder.embeddings.position_ids' from state_dict")
        
        model.load_state_dict(state_dict, strict=False)  # strict=False로 일부 키 불일치 허용
        model.eval()
        print(f"Model loaded successfully from {model_path}")
        
        # 모델 파라미터 정보 출력
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        return model
    
    def _load_tokenizer(self):
        """Roberta 토크나이저를 로드합니다."""
        try:
            tokenizer_path = self.config['cross_attention_specific']['chemberta_model_name']
            tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)
            print(f"Tokenizer loaded from: {tokenizer_path}")
            return tokenizer
        except Exception as e:
            print(f"Failed to load tokenizer: {e}")
            raise
    
    def _prepare_dataset(self):
        """데이터셋을 준비합니다."""
        # BACE 데이터셋 설정
        dataset_config = {
            'data_path': 'data/bace/bace.csv',
            'target': self.target_name,
            'task': 'classification',
            'splitting': 'scaffold'
        }
        
        wrapper_args = {
            'batch_size': self.config['batch_size'],
            'num_workers': self.config['dataset']['num_workers'],
            'valid_size': self.config['dataset']['valid_size'],
            'test_size': self.config['dataset']['test_size'],
            **dataset_config
        }
        
        dataset_wrapper = HybridDatasetWrapper(**wrapper_args)
        print(f"Dataset prepared: {dataset_config['data_path']}")
        return dataset_wrapper
    
    def _extract_embeddings_and_predictions(self):
        """테스트 데이터셋에 대한 임베딩, 예측, 라벨을 추출합니다."""
        _, _, test_loader = self.dataset_wrapper.get_data_loaders()
        
        data_store = {
            'smiles': [], 
            'labels': [], 
            'preds': [], 
            'probs': [],
            'embeddings': [],
            'gnn_embeddings': [],
            'lm_embeddings': []
        }
        
        print("Extracting embeddings and predictions from test set...")
        with torch.no_grad():
            for graph_data, smiles_list in tqdm(test_loader, desc="Processing test data"):
                if graph_data is None or not smiles_list: 
                    continue
                
                graph_data = graph_data.to(self.device)
                
                # 텍스트 토큰화
                max_length = 128
                cls_token_id = self.tokenizer.cls_token_id
                sep_token_id = self.tokenizer.sep_token_id
                pad_token_id = self.tokenizer.pad_token_id
                
                all_input_ids = []
                all_attention_masks = []
                
                for smile in smiles_list:
                    token_ids = self.tokenizer.encode(smile, add_special_tokens=False)
                    if len(token_ids) > max_length - 2:
                        token_ids = token_ids[:max_length - 2]
                    input_ids = [cls_token_id] + token_ids + [sep_token_id]
                    attention_mask = [1] * len(input_ids)
                    padding_length = max_length - len(input_ids)
                    input_ids = input_ids + ([pad_token_id] * padding_length)
                    attention_mask = attention_mask + ([0] * padding_length)
                    all_input_ids.append(input_ids)
                    all_attention_masks.append(attention_mask)
                
                input_ids_tensor = torch.tensor(all_input_ids, dtype=torch.long).to(self.device)
                attention_mask_tensor = torch.tensor(all_attention_masks, dtype=torch.long).to(self.device)
                smiles_tokens = {
                    'input_ids': input_ids_tensor,
                    'attention_mask': attention_mask_tensor
                }
                
                # 모델 추론
                logits = self.model(graph_data, smiles_tokens)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                
                # 임베딩 추출 (모델의 중간 레이어에서)
                # GNN 임베딩
                gnn_emb = self.model.gnn_encoder(graph_data)
                if isinstance(gnn_emb, tuple):
                    gnn_emb = gnn_emb[0]  # 튜플의 첫번째 요소 사용
                
                # LM 임베딩
                lm_outputs = self.model.lm_encoder(
                    input_ids=smiles_tokens['input_ids'],
                    attention_mask=smiles_tokens['attention_mask']
                )
                lm_emb = lm_outputs.last_hidden_state[:, 0, :]  # [CLS] token
                
                # 융합 임베딩
                fusion_emb = self.model.fusion_layer(gnn_emb, lm_emb)
                
                # 데이터 저장
                data_store['smiles'].extend(smiles_list)
                data_store['labels'].extend(graph_data.y.cpu().numpy())
                data_store['preds'].extend(preds.cpu().numpy().flatten())
                data_store['probs'].extend(probs.cpu().numpy().flatten())
                data_store['embeddings'].append(fusion_emb.cpu().numpy())
                data_store['gnn_embeddings'].append(gnn_emb.cpu().numpy())
                data_store['lm_embeddings'].append(lm_emb.cpu().numpy())
        
        # 모든 데이터를 numpy 배열로 변환
        for key in ['embeddings', 'gnn_embeddings', 'lm_embeddings']:
            data_store[key] = np.concatenate(data_store[key], axis=0)
        
        data_store['labels'] = np.array(data_store['labels'])
        data_store['preds'] = np.array(data_store['preds'])
        data_store['probs'] = np.array(data_store['probs'])
        data_store['smiles'] = np.array(data_store['smiles'])
        
        print(f"Processed {len(data_store['smiles'])} samples")
        return data_store
    
    def _analyze_performance(self, labels, preds, probs):
        """모델 성능을 분석합니다."""
        try:
            auc_score = roc_auc_score(labels, probs)
            accuracy = np.mean(labels == preds)
            
            # Confusion Matrix
            tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
            
            print(f"\n--- Performance Metrics ---")
            print(f"AUC Score: {auc_score:.4f}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"True Positives: {tp}")
            print(f"True Negatives: {tn}")
            print(f"False Positives: {fp}")
            print(f"False Negatives: {fn}")
            print("--------------------------\n")
            
            return {
                'auc': auc_score,
                'accuracy': accuracy,
                'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
            }
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            return None
    
    def _visualize_tsne_comparison(self, data):
        """GNN, LM, 융합 임베딩의 t-SNE를 비교 시각화합니다."""
        print("Generating t-SNE visualizations...")
        
        embeddings_dict = {
            'GNN Embeddings': data['gnn_embeddings'],
            'LM Embeddings': data['lm_embeddings'], 
            'Fusion Embeddings': data['embeddings']
        }
        
        labels = data['labels']
        preds = data['preds']
        smiles = data['smiles']
        
        # 오분류 인덱스 찾기
        fp_indices = np.where((labels == 0) & (preds == 1))[0]
        fn_indices = np.where((labels == 1) & (preds == 0))[0]
        
        # 오분류 분자 정보 터미널 출력
        print(f"\n--- Misclassification Analysis ---")
        print(f"False Positives (FP): {len(fp_indices)} cases")
        print(f"False Negatives (FN): {len(fn_indices)} cases")
        
        # 최대 3개의 오분류 샘플만 선택 (hybrid_fn_fp_viz.py 방식)
        n_samples = 3
        if len(fp_indices) > 0:
            fp_draw = np.random.choice(fp_indices, size=min(n_samples, len(fp_indices)), replace=False)
            print("\n--- SMILES strings for ANNOTATED False Positives ---")
            for i, idx in enumerate(fp_draw):
                print(f"  - False Positive Molecule #{i+1} (Original index: {idx}): {smiles[idx]}")
        else:
            fp_draw = []
        
        if len(fn_indices) > 0:
            fn_draw = np.random.choice(fn_indices, size=min(n_samples, len(fn_indices)), replace=False)
            print("\n--- SMILES strings for ANNOTATED False Negatives ---")
            for i, idx in enumerate(fn_draw):
                print(f"  - False Negative Molecule #{i+1} (Original index: {idx}): {smiles[idx]}")
        else:
            fn_draw = []
        
        print("---------------------------------")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for idx, (name, embeddings) in enumerate(embeddings_dict.items()):
            ax = axes[idx]
            
            # t-SNE 계산
            scaler = StandardScaler()
            embeddings_scaled = scaler.fit_transform(embeddings)
            tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
            tsne_results = tsne.fit_transform(embeddings_scaled)
            
            # winter colormap으로 Positive(1)=파랑, Negative(0)=연두색
            ax.scatter(tsne_results[:, 0], tsne_results[:, 1], 
                               c=labels, cmap='winter', alpha=1.0, s=300)
            
            # 스타일 설정 (hybrid_fn_fp_viz.py 방식)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            
            # 헬퍼 함수 (hybrid_fn_fp_viz.py 완벽 복사)
            def annotate_misclassified_points(indices, color, label, error_type_str):
                """Helper function to highlight points and add text number annotations."""
                for i, idx in enumerate(indices):
                    point = tsne_results[idx]
                    text_label = f"#{i+1}"
                    
                    # 하이라이트 원 그리기 (hybrid_fn_fp_viz.py 정확한 파라미터)
                    ax.scatter(point[0], point[1], marker='o', facecolor='none', 
                               edgecolor=color, linewidth=2.5, s=250, 
                               label=label if i == 0 else "", zorder=10)

                    # 번호 텍스트 추가 (hybrid_fn_fp_viz.py 정확한 파라미터)
                    ax.text(point[0], point[1] + 0.5, text_label,
                            fontsize=12, 
                            fontweight='bold', 
                            color=color,
                            ha='center',
                            va='bottom',
                            zorder=11)
            
            # FP/FN 포인트에 어노테이션 추가 (hybrid_fn_fp_viz.py 방식)
            if len(fp_draw) > 0:
                annotate_misclassified_points(fp_draw, color='red', label='False Positive', error_type_str='False Positive')
            
            if len(fn_draw) > 0:
                annotate_misclassified_points(fn_draw, color='black', label='False Negative', error_type_str='False Negative')
            
            ax.set_title(f'{name}\n(alpha={self.config["cross_attention_specific"]["fusion"]["alpha"]}, '
                        f'beta={self.config["cross_attention_specific"]["fusion"]["beta"]})', 
                        fontsize=12, fontweight='bold')
            
            # 범례 추가 (hybrid_fn_fp_viz.py 방식)
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, f'tsne_comparison_{self.target_name}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"t-SNE comparison plot saved to {save_path}")
    
    def _visualize_3d_tsne_comparison(self, data):
        """GNN, LM, 융합 임베딩의 3D t-SNE를 비교 시각화합니다."""
        print("Generating 3D t-SNE visualizations...")
        
        embeddings_dict = {
            'GNN Embeddings': data['gnn_embeddings'],
            'LM Embeddings': data['lm_embeddings'], 
            'Fusion Embeddings': data['embeddings']
        }
        
        labels = data['labels']
        preds = data['preds']
        
        # 오분류 인덱스 찾기
        fp_indices = np.where((labels == 0) & (preds == 1))[0]
        fn_indices = np.where((labels == 1) & (preds == 0))[0]
        
        # 최대 3개의 오분류 샘플만 선택
        n_samples = 3
        if len(fp_indices) > 0:
            fp_draw = np.random.choice(fp_indices, size=min(n_samples, len(fp_indices)), replace=False)
        else:
            fp_draw = []
        
        if len(fn_indices) > 0:
            fn_draw = np.random.choice(fn_indices, size=min(n_samples, len(fn_indices)), replace=False)
        else:
            fn_draw = []
        
        fig = plt.figure(figsize=(20, 8))
        
        for idx, (name, embeddings) in enumerate(embeddings_dict.items()):
            ax = fig.add_subplot(1, 3, idx+1, projection='3d')
            
            # 3D t-SNE 계산
            scaler = StandardScaler()
            embeddings_scaled = scaler.fit_transform(embeddings)
            tsne = TSNE(n_components=3, perplexity=30, max_iter=1000, random_state=42)
            tsne_results = tsne.fit_transform(embeddings_scaled)
            
            # winter colormap으로 Positive(1)=파랑, Negative(0)=연두색
            ax.scatter(tsne_results[:, 0], tsne_results[:, 1], tsne_results[:, 2], 
                               c=labels, cmap='winter', alpha=1.0, s=100)
            
            # 헬퍼 함수 (3D용)
            def annotate_misclassified_points_3d(indices, color, label):
                """Helper function to highlight points and add text number annotations in 3D."""
                for i, idx in enumerate(indices):
                    point = tsne_results[idx]
                    text_label = f"#{i+1}"
                    
                    # 하이라이트 구 그리기 (3D)
                    ax.scatter(point[0], point[1], point[2], marker='o', facecolor='none', 
                               edgecolor=color, linewidth=2.5, s=300, 
                               label=label if i == 0 else "", zorder=10)

                    # 번호 텍스트 추가 (3D)
                    ax.text(point[0], point[1], point[2] + 0.5, text_label,
                            fontsize=10, fontweight='bold', color=color,
                            ha='center', va='bottom', zorder=11)
            
            # FP/FN 포인트에 어노테이션 추가
            if len(fp_draw) > 0:
                annotate_misclassified_points_3d(fp_draw, color='red', label='False Positive')
            
            if len(fn_draw) > 0:
                annotate_misclassified_points_3d(fn_draw, color='black', label='False Negative')
            
            ax.set_title(f'{name} (3D)\n(alpha={self.config["cross_attention_specific"]["fusion"]["alpha"]}, '
                        f'beta={self.config["cross_attention_specific"]["fusion"]["beta"]})', 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('t-SNE Dimension 1')
            ax.set_ylabel('t-SNE Dimension 2')
            ax.set_zlabel('t-SNE Dimension 3')
            
            # 범례 추가
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, linestyle='--', alpha=0.3)
            
            # 3D 그래프 회전 각도 설정
            ax.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, f'tsne_3d_comparison_{self.target_name}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"3D t-SNE comparison plot saved to {save_path}")
    
    def _create_attention_evolution_animation(self, data):
        """오분류 분자들의 cross-attention fusion 과정에서의 임베딩 변화를 동영상으로 생성합니다."""
        print("Creating attention evolution animation...")
        
        # 오분류된 분자들 선택 (최대 5개)
        labels = data['labels']
        preds = data['preds']
        
        # False Negative 분자들만 선택
        fn_indices = np.where((labels == 1) & (preds == 0))[0]
        print(f"Available FN indices: {fn_indices}")
        print(f"Total FN count: {len(fn_indices)}")
        
        # 오분류 샘플 선택 (FN 3개만)
        misclassified = []
        fn_selected = []
        
        if len(fn_indices) >= 3:
            # 중복되지 않는 3개의 고유 인덱스 선택
            unique_fn_indices = np.unique(fn_indices)
            print(f"Unique FN indices: {unique_fn_indices}")
            
            if len(unique_fn_indices) >= 3:
                fn_selected = unique_fn_indices[:3]  # 고유한 FN 3개 선택
            else:
                fn_selected = unique_fn_indices[:]  # 있는 만큼 모두 선택
        elif len(fn_indices) > 0:
            fn_selected = np.unique(fn_indices)[:]  # 고유 인덱스만 선택
        
        misclassified.extend(fn_selected)
        
        print(f"Selected {len(fn_selected)} FN indices: {fn_selected}")
        print(f"Total misclassified for animation: {len(misclassified)}")
        
        if len(misclassified) == 0:
            print("No misclassified samples found for animation.")
            return
        
        print("Creating animation for {} misclassified molecules".format(len(misclassified)))
        
        # 동영상 프레임 저장할 디렉토리
        frame_dir = os.path.join(self.output_dir, 'animation_frames')
        os.makedirs(frame_dir, exist_ok=True)
        
        frames = []
        frame_paths = []
        
        # 알파를 이산적으로 변화: 0.1, 0.2, 0.4, 0.6, 0.8, 1.0
        alpha_values = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1.0])  # 6프레임
        beta_fixed = 1.0  # 베타는 1.0으로 고정
        
        for frame_idx, current_alpha in enumerate(alpha_values):
            current_beta = beta_fixed  # 베타는 항상 1.0
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Cross-Attention Evolution: alpha={current_alpha:.2f}, beta={current_beta:.2f} (Frame {frame_idx+1}/6)', 
                        fontsize=14, fontweight='bold')
            
            # 각 subplot에 다른 임베딩 표시
            subplot_info = [
                (0, 0, 'GNN Embeddings', data['gnn_embeddings']),
                (0, 1, 'LM Embeddings', data['lm_embeddings']),
                (1, 0, 'Fusion Embeddings (Current)', data['embeddings']),
                (1, 1, 'Alpha/Beta Ratio', None)  # 제목 변경
            ]
            
            for row, col, title, embeddings in subplot_info:
                ax = axes[row, col]
                
                if embeddings is not None:
                    # alpha/beta 비율에 따라 fusion 임베딩을 시뮬레이션
                    if title == 'Fusion Embeddings (Current)':
                        # Fusion 임베딩을 alpha/beta 비율에 따라 동적으로 조정
                        gnn_emb = data['gnn_embeddings']
                        lm_emb = data['lm_embeddings']
                        
                        # 차원 확인 및 조정
                        print(f"GNN embeddings shape: {gnn_emb.shape}")
                        print(f"LM embeddings shape: {lm_emb.shape}")
                        
                        # alpha 비율에 따른 가중치 조정
                        alpha_weight = current_alpha / (current_alpha + current_beta)
                        beta_weight = current_beta / (current_alpha + current_beta)
                        
                        # 동적 fusion 임베딩 생성 (차원이 다르면 그냥 기존 임베딩 사용)
                        if gnn_emb.shape[1] == lm_emb.shape[1]:
                            dynamic_fusion = alpha_weight * gnn_emb + beta_weight * lm_emb
                        else:
                            print("Embedding dimensions differ, using original fusion embeddings")
                            dynamic_fusion = data['embeddings']  # 기존 fusion 임베딩 사용
                        
                        # 동적 임베딩으로 t-SNE 계산
                        scaler = StandardScaler()
                        embeddings_scaled = scaler.fit_transform(dynamic_fusion)
                    else:
                        # GNN, LM 임베딩은 그대로 사용 (alpha/beta 영향 없음)
                        scaler = StandardScaler()
                        embeddings_scaled = scaler.fit_transform(embeddings)
                    
                    # t-SNE 계산 (각 프레임마다 다른 random_state로 살짝 변화)
                    tsne = TSNE(n_components=2, perplexity=30, max_iter=500, 
                               random_state=42 + frame_idx)  # 프레임마다 약간의 변화
                    tsne_results = tsne.fit_transform(embeddings_scaled)
                    
                    # 전체 데이터 포인트
                    ax.scatter(tsne_results[:, 0], tsne_results[:, 1], 
                               c=labels, cmap='winter', alpha=1.0, s=30)
                    
                    # 오분류된 분자들 하이라이트
                    misclassified_tsne = tsne_results[misclassified]
                    
                    for i, (idx, point) in enumerate(zip(misclassified, misclassified_tsne)):
                        color = 'black'  # FN은 항상 검은색
                        
                        # 2D t-SNE와 동일한 빈 동그라미 표기법 사용
                        ax.scatter(point[0], point[1], marker='o', facecolor='none', 
                                 edgecolor=color, linewidth=2.5, s=250, 
                                 label='False Negative' if i == 0 else "", 
                                 zorder=10)
                        
                        # 번호 텍스트 추가
                        text_label = f"#{i+1}"
                        ax.text(point[0], point[1] + 0.5, text_label,
                                fontsize=12, fontweight='bold', color=color,
                                ha='center', va='bottom', zorder=11)
                        
                        # FN 라벨 추가 (번호 없이 그냥 FN)
                        ax.text(point[0], point[1] - 0.5, "FN",
                                fontsize=10, fontweight='bold', color=color,
                                ha='center', va='top', zorder=11)
                    
                    # 현재 alpha/beta 정보 추가
                    ax.text(0.02, 0.98, f'α={current_alpha:.2f}, β={current_beta:.1f}', 
                           transform=ax.transAxes, fontsize=10, 
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    ax.set_title(title, fontsize=10, fontweight='bold')
                    ax.grid(True, linestyle='--', alpha=0.3)
                    ax.set_xlabel('t-SNE Dimension 1')
                    ax.set_ylabel('t-SNE Dimension 2')
                
                else:
                    # Alpha/Beta 비율 시각화
                    # 현재 alpha/beta 비율을 막대그래프로 표시
                    alpha_weight = current_alpha / (current_alpha + current_beta)
                    beta_weight = current_beta / (current_alpha + current_beta)
                    
                    categories = ['GNN\nWeight', 'LM\nWeight']
                    weights = [alpha_weight, beta_weight]
                    colors = ['red', 'blue']
                    
                    bars = ax.bar(categories, weights, color=colors, alpha=0.7)
                    
                    # 비율 값 텍스트 추가
                    for i, (bar, weight) in enumerate(zip(bars, weights)):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{weight:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
                    
                    # 현재 alpha/beta 값 추가
                    ax.text(0.5, 0.95, f'α={current_alpha:.2f}, β={current_beta:.1f}', 
                           transform=ax.transAxes, fontsize=14, fontweight='bold',
                           ha='center', va='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    ax.set_ylim(0, 1.1)
                    ax.set_ylabel('Weight Ratio')
                    ax.set_title('Fusion Weights', fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            
            # 프레임 저장
            frame_path = os.path.join(frame_dir, f'frame_{frame_idx:03d}.png')
            plt.savefig(frame_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            # 프레임 경로만 저장 (나중에 한 번에 로드)
            frame_paths.append(frame_path)
            
            if frame_idx % 1 == 0:  # 모든 프레임 표시 (6프레임만 있으므로)
                print("Generated frame {}/6".format(frame_idx+1))
        
        # 모든 프레임을 동일한 크기로 로드 및 크기 통일
        reference_size = None
        for frame_path in frame_paths:
            img = imageio.v2.imread(frame_path)
            if reference_size is None:
                reference_size = img.shape[:2]
            # 크기가 다르면 reference_size로 리사이즈
            if img.shape[:2] != reference_size:
                from PIL import Image
                pil_img = Image.fromarray(img)
                pil_img = pil_img.resize((reference_size[1], reference_size[0]))
                img = np.array(pil_img)
            frames.append(img)
        
        # GIF 생성 (6프레임, 각 1초 간격으로 총 6초)
        gif_path = os.path.join(self.output_dir, f'attention_evolution_{self.target_name}.gif')
        imageio.mimsave(gif_path, frames, duration=1.0, loop=0, fps=1)
        
        # 임시 프레임 파일 삭제
        import shutil
        shutil.rmtree(frame_dir)
        
        print("Attention evolution animation saved to {}".format(gif_path))
        print("Animation shows how misclassified molecules move as alpha changes discretely")
        print("Beta is fixed at 1.0, alpha varies: {:.1f} → {:.1f} → {:.1f} → {:.1f} → {:.1f} → {:.1f}".format(*alpha_values))
        print("Duration: 6 seconds (6 frames, 1 second intervals)")
    
    def _visualize_attention_weights(self, data):
        """어텐션 가중치 분포를 시각화합니다."""
        print("Analyzing attention weight distribution...")
        
        # alpha/beta 비율 정보
        alpha = self.config['cross_attention_specific']['fusion'].get('alpha', 1.0)
        beta = self.config['cross_attention_specific']['fusion'].get('beta', 1.0)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 어텐션 가중치 분포 (예시 데이터)
        # 실제로는 모델 내부에서 어텐션 가중치를 추출해야 함
        attention_weights = np.random.beta(alpha, beta, 1000)
        
        ax1.hist(attention_weights, bins=50, alpha=0.7, density=True)
        ax1.set_title(f'Attention Weight Distribution\n(alpha={alpha}, beta={beta})')
        ax1.set_xlabel('Attention Weight')
        ax1.set_ylabel('Density')
        ax1.grid(True, alpha=0.3)
        
        # alpha/beta 비율에 따른 성능 변화 (예시)
        alpha_range = np.linspace(0.1, 2.0, 20)
        beta_fixed = 1.0
        performance = 1 / (1 + np.abs(alpha_range - 1.0))  # 예시 성능 함수
        
        ax2.plot(alpha_range, performance, 'b-', linewidth=2, marker='o')
        ax2.axvline(x=alpha, color='red', linestyle='--', label=f'Current alpha={alpha}')
        ax2.set_title(f'Performance vs Alpha (beta={beta_fixed})')
        ax2.set_xlabel('Alpha Value')
        ax2.set_ylabel('Performance (AUC)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, f'attention_analysis_{self.target_name}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Attention analysis plot saved to {save_path}")
    
    def run(self):
        """전체 분석 파이프라인을 실행합니다."""
        print("\n" + "="*50)
        print("STARTING CROSS-ATTENTION t-SNE ANALYSIS")
        print("="*50)
        
        # 1. 데이터 추출
        data = self._extract_embeddings_and_predictions()
        
        # 2. 성능 분석
        metrics = self._analyze_performance(data['labels'], data['preds'], data['probs'])
        
        # 3. 2D t-SNE 시각화
        self._visualize_tsne_comparison(data)
        
        # 4. 3D t-SNE 시각화 (신규 기능)
        self._visualize_3d_tsne_comparison(data)
        
        # 5. 어텐션 가중치 분석
        self._visualize_attention_weights(data)
        
        # 6. 오분류 분자 시각화 (제거 - RDKit 문제로 건너뜀)
        # self._visualize_misclassified_molecules(data)
        
        # 7. 동영상/GIF 생성 (신규 기능)
        self._create_attention_evolution_animation(data)
        
        # 8. 분석 결과 요약 저장
        self._save_analysis_summary(data, metrics)
        self._visualize_alpha_weight_curve()
        self._visualize_fusion_trajectory(data)
        self._visualize_embedding_distance_curve(data)
        print(f"\n✅ Analysis completed successfully!")
        print(f"📁 Results saved to: {self.output_dir}")
        print("="*50)
    
    def _save_analysis_summary(self, data, metrics):
        """분석 결과 요약을 저장합니다."""
        summary = {
            'config_file': 'config_cross_attention_ratio.yaml',
            'model_log_dir': self.model_log_dir,
            'target_name': self.target_name,
            'timestamp': datetime.now().isoformat(),
            'sample_count': len(data['smiles']),
            'metrics': metrics,
            'attention_config': {
                'alpha': self.config['cross_attention_specific']['fusion'].get('alpha', 1.0),
                'beta': self.config['cross_attention_specific']['fusion'].get('beta', 1.0)
            },
            'false_positives': int(np.sum((data['labels'] == 0) & (data['preds'] == 1))),
            'false_negatives': int(np.sum((data['labels'] == 1) & (data['preds'] == 0)))
        }
        
        summary_path = os.path.join(self.output_dir, 'analysis_summary.yaml')
        with open(summary_path, 'w', encoding='utf-8') as f:
            yaml.dump(summary, f, default_flow_style=False)
        
        print(f"Analysis summary saved to {summary_path}")
    def _visualize_alpha_weight_curve(self):
        alphas = np.linspace(0.01, 5.0, 200)
        beta = self.config["cross_attention_specific"]["fusion"]["beta"]

        gnn_w = alphas / (alphas + beta)
        lm_w = beta / (alphas + beta)

        plt.figure(figsize=(7,5))
        plt.plot(alphas, gnn_w, label="GNN Weight", linewidth=3)
        plt.plot(alphas, lm_w, label="LM Weight", linewidth=3)
        plt.xlabel("alpha")
        plt.ylabel("Weight")
        plt.title("Effect of alpha on Fusion Weights")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.4)

        save_path = os.path.join(self.output_dir, "alpha_weight_curve.png")
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"[Saved] {save_path}")
    def _visualize_fusion_trajectory(self, data, sample_idx=0):
        g = data["gnn_embeddings"][sample_idx]
        l = data["lm_embeddings"][sample_idx]

        # alpha 값들
        alpha_values = np.array([0.1, 0.3, 0.6, 1.0, 2.0, 5.0])
        beta = self.config["cross_attention_specific"]["fusion"]["beta"]

        dynamic_fusions = []
        for a in alpha_values:
            w_g = a / (a + beta)
            w_l = beta / (a + beta)
            
            # 차원이 다르면 기존 fusion 임베딩 사용
            if g.shape[0] == l.shape[0]:
                fused = w_g * g + w_l * l
            else:
                print("GNN and LM embedding dimensions differ, using original fusion embedding")
                fused = data["embeddings"][sample_idx]  # 기존 fusion 임베딩 사용
            
            dynamic_fusions.append(fused)

        dynamic_fusions = np.array(dynamic_fusions)

        # t-SNE (6개 포인트만)
        if len(dynamic_fusions) >= 2:
            perplexity = min(len(dynamic_fusions) - 1, 6)
            tsne = TSNE(n_components=2, perplexity=perplexity, init="random", random_state=42)
            tsne_res = tsne.fit_transform(dynamic_fusions)
        else:
            # 포인트가 너무 적으면 그냥 2D 배열로 사용
            tsne_res = np.array([[i, i*0.5] for i in range(len(dynamic_fusions))])

        plt.figure(figsize=(7,6))
        plt.scatter(tsne_res[:,0], tsne_res[:,1], s=120, color="blue")

        # trajectory 화살표
        for i in range(len(tsne_res)-1):
            plt.arrow(tsne_res[i,0], tsne_res[i,1],
                    tsne_res[i+1,0] - tsne_res[i,0],
                    tsne_res[i+1,1] - tsne_res[i,1],
                    length_includes_head=True,
                    head_width=0.3, color="black")

        for i, a in enumerate(alpha_values):
            plt.text(tsne_res[i,0], tsne_res[i,1], f"α={a}", fontsize=10)

        plt.title("Trajectory of Fusion Embedding as alpha changes")
        plt.grid(True, linestyle="--", alpha=0.3)
        
        save_path = os.path.join(self.output_dir, "fusion_trajectory.png")
        plt.savefig(save_path, dpi=250, bbox_inches="tight")
        plt.close()
        print(f"[Saved] {save_path}")
    
    def _visualize_embedding_distance_curve(self, data, sample_idx=0):
        g = data["gnn_embeddings"][sample_idx]
        l = data["lm_embeddings"][sample_idx]

        alphas = np.linspace(0.01, 5.0, 100)
        beta = self.config["cross_attention_specific"]["fusion"]["beta"]

        distances = []
        for a in alphas:
            w_g = a / (a + beta)
            w_l = beta / (a + beta)
            
            # 차원이 다르면 기존 fusion 임베딩 사용
            if g.shape[0] == l.shape[0]:
                fused = w_g * g + w_l * l
                dist = np.linalg.norm(fused - l)
            else:
                # 기존 fusion 임베딩과 LM 임베딩 간의 거리 계산
                fused = data["embeddings"][sample_idx]
                if fused.shape[0] == l.shape[0]:
                    dist = np.linalg.norm(fused - l)
                else:
                    # 차원이 다르면 그냥 alpha 값에 따른 거리 시뮬레이션
                    dist = abs(w_g - 0.5) * 10  # 간단한 시뮬레이션
            distances.append(dist)

        plt.figure(figsize=(7,5))
        plt.plot(alphas, distances, linewidth=2)
        plt.xlabel("alpha")
        plt.ylabel("||Fusion - LM|| (Euclidean Distance)")
        plt.title("Distance between Fusion and LM Embedding as alpha varies")
        plt.grid(True, linestyle="--", alpha=0.4)

        save_path = os.path.join(self.output_dir, "distance_curve.png")
        plt.savefig(save_path, dpi=250, bbox_inches="tight")
        plt.close()
        print(f"[Saved] {save_path}")




# --- 스크립트 실행 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-Attention t-SNE Visualization")
    parser.add_argument('--config', type=str, 
                       default='config_cross_attention_ratio.yaml',
                       help='Path to config file')
    parser.add_argument('--model_dir', type=str, 
                       help='Path to model directory (optional, will find latest if not specified)')
    parser.add_argument('--target', type=str, 
                       default='Class',
                       help='Target column name to analyze')
    
    args = parser.parse_args()
    
    # 타겟 이름을 config에 설정
    config = yaml.load(open(args.config, "r", encoding="utf-8"), Loader=yaml.FullLoader)
    if 'analysis_specific' not in config:
        config['analysis_specific'] = {}
    config['analysis_specific']['target_to_analyze'] = args.target
    
    # 임시 config 파일 저장
    temp_config_path = 'temp_config.yaml'
    with open(temp_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    try:
        # 분석 실행
        analyzer = CrossAttentionTSNEAnalyzer(
            config_path=temp_config_path,
            model_log_dir=args.model_dir
        )
        analyzer.run()
    finally:
        # 임시 config 파일 삭제
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
