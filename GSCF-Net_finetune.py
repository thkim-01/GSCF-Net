# attention_ratio_finetune.py
"""
[alpha/beta 비율 조절 실험 전용 스크립트]
이 스크립트는 `models/attention_ratio.py`에 정의된 모델을 사용하여
어텐션의 Key/Value 비율 조정 실험을 수행합니다.

실행 예시:
$ python attention_ratio_finetune.py --dataset BBBP --config config_attention_ratio.yaml
"""

# -------------------------------------------------------------------
# 0. Import 필수 라이브러리 (기존과 동일)
# -------------------------------------------------------------------
import os
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
import subprocess
import argparse
import time

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from transformers import RobertaModel, RobertaTokenizer

# --- 사용할 사용자 정의 클래스 import ---
from dataset.hybrid_dataset import HybridDatasetWrapper
#  [수정 1/3] 사용할 모델 클래스를 attention_ratio.py에서 가져옵니다.
from models.cross_attention_ratio import HybridCrossAttentionModel
from sklearn.metrics import roc_auc_score

# -------------------------------------------------------------------
# 2. StandaloneTrainer 클래스 (내부 로직은 기존과 완전히 동일)
# -------------------------------------------------------------------
class StandaloneTrainer:
    def __init__(self, dataset_wrapper, config):
        self.config = config
        self.device = self._get_device()

        self.train_loader, self.valid_loader, self.test_loader = dataset_wrapper.get_data_loaders()

        # HybridCrossAttentionModel을 호출하면 `models.attention_ratio` 버전이 사용됩니다.
        self.model = HybridCrossAttentionModel(config).to(self.device)
        self.optimizer = self._create_optimizer()
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config['epochs'], eta_min=1e-6)
        self.criterion = nn.BCEWithLogitsLoss()
        
        # ★★★★★ [수정 2/3] TensorBoard 로그 저장 경로를 구분합니다. ★★★★★
        log_dir_name = f"{config['task_name']}_{datetime.now().strftime('%b%d_%H-%M-%S')}"
        self.log_dir = os.path.join('runs_ratio', log_dir_name) # 'runs_ca' -> 'runs_ratio'
        self.writer = SummaryWriter(self.log_dir)
        print(f"Tensorboard log will be saved to: {self.log_dir}")
        
        try:
            from transformers import RobertaTokenizer
            tokenizer_path = config['cross_attention_specific']['chemberta_model_name']
            self.tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)
            print(f"Successfully loaded RobertaTokenizer from: {tokenizer_path}")
        except Exception as e:
            print(f"[FATAL ERROR] Failed to load tokenizer from {tokenizer_path}.")
            raise e

    # _get_device, _create_optimizer, _step, train, _evaluate 메서드는
    # 기존 `cross_attention_finetune.py`와 완전히 동일하므로 코드를 그대로 붙여넣습니다.
    # ... (이하 모든 메서드 코드는 변경 없이 동일) ...
    def _get_device(self):
        if torch.cuda.is_available() and 'gpu' in self.config:
            return f"cuda:{self.config['gpu']}"
        return "cpu"
    def _create_optimizer(self):
        gnn_params = self.model.gnn_encoder.parameters()
        lm_params = self.model.lm_encoder.parameters()
        head_params = list(self.model.fusion_layer.parameters()) + list(self.model.classifier.parameters())
        grouped_params = [
            {'params': gnn_params, 'lr': float(self.config['init_base_lr'])},
            {'params': lm_params, 'lr': float(self.config['cross_attention_specific']['chemberta_lr'])},
            {'params': head_params, 'lr': float(self.config['init_lr'])}
        ]
        optimizer = torch.optim.AdamW(
            params=grouped_params,
            weight_decay=float(self.config['weight_decay']),
            betas=(0.9, 0.999),
            eps=1e-8
        )
        print("Optimizer configured with differential learning rates.")
        return optimizer
    def _step(self, data):
        graph_data, smiles_list = data
        if graph_data is None or not smiles_list: return None
        graph_data = graph_data.to(self.device)
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
        logits = self.model(graph_data, smiles_tokens)
        labels = graph_data.y.view(-1, 1).float()
        loss = self.criterion(logits, labels)
        return loss, logits, labels
    def train(self):
        best_valid_score = 0
        best_test_score = 0
        for epoch in range(1, self.config['epochs'] + 1):
            self.model.train()
            train_loss = 0
            train_loop = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config['epochs']} [Train]", unit="batch")
            for data in train_loop:
                self.optimizer.zero_grad()
                step_result = self._step(data)
                if step_result is None: continue
                loss, _, _ = step_result
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                train_loop.set_postfix(loss=loss.item())
            self.scheduler.step()
            avg_train_loss = train_loss / len(self.train_loader)
            self.writer.add_scalar('Loss/train', avg_train_loss, epoch)
            if epoch % self.config['eval_every_n_epochs'] == 0:
                valid_score = self._evaluate(self.valid_loader, 'Valid', epoch)
                self.writer.add_scalar(f"Score/roc_auc_valid", valid_score, epoch)
                if valid_score > best_valid_score:
                    best_valid_score = valid_score
                    best_test_score = self._evaluate(self.test_loader, 'Test', epoch)
                    self.writer.add_scalar(f"Score/roc_auc_test_at_best_valid", best_test_score, epoch)
                    print(f"🚀 Epoch {epoch}: New best valid score: {best_valid_score:.4f}, Corresponding test score: {best_test_score:.4f}")
                    save_path = os.path.join(self.writer.log_dir, 'best_model.pth')
                    torch.save(self.model.state_dict(), save_path)
                    print(f"Best model saved to {save_path}")
        print("Training finished.")
        self.writer.close()
        return best_test_score
    def _evaluate(self, loader, split_name, epoch):
        self.model.eval()
        all_preds, all_labels = [], []
        eval_loop = tqdm(loader, desc=f"Epoch {epoch}/{self.config['epochs']} [{split_name}]", leave=False, unit="batch")
        with torch.no_grad():
            for data in eval_loop:
                step_result = self._step(data)
                if step_result is None: continue
                _, logits, labels = step_result
                preds = torch.sigmoid(logits)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        if not all_labels: return 0.0
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        try:
            if len(np.unique(all_labels)) < 2: score = 0.5
            else: score = roc_auc_score(all_labels, all_preds)
        except Exception: score = 0.0
        return score

# -------------------------------------------------------------------
# 3. 메인 실행 스크립트 로직 (기존과 거의 동일)
# -------------------------------------------------------------------

# DATASET_CONFIGS, get_git_hash, run_single_target_experiment 함수는 기존과 완전히 동일
# ... (코드 생략) ...
DATASET_CONFIGS = {
    'BBBP': {'data_path': 'data/bbbp/BBBP.csv', 'target': 'p_np', 'task': 'classification'},
    'Tox21': {'data_path': 'data/tox21/tox21.csv', 'target': ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'], 'task': 'classification'},
    'ClinTox': {'data_path': 'data/clintox/clintox.csv', 'target': ['CT_TOX','FDA_APPROVED'], 'task': 'classification'},
    'HIV': {'data_path': 'data/hiv/HIV.csv', 'target': 'HIV_active', 'task': 'classification'},
    'BACE': {'data_path': 'data/bace/bace.csv', 'target': 'Class', 'task': 'classification'},
    'SIDER': {'data_path': 'data/sider/sider.csv', 'target': ["Hepatobiliary disorders", "Metabolism and nutrition disorders", "Product issues", "Eye disorders", "Investigations", "Musculoskeletal and connective tissue disorders", "Gastrointestinal disorders", "Social circumstances", "Immune system disorders", "Reproductive system and breast disorders", "Neoplasms benign, malignant and unspecified (incl cysts and polyps)", "General disorders and administration site conditions", "Endocrine disorders", "Surgical and medical procedures", "Vascular disorders", "Blood and lymphatic system disorders", "Skin and subcutaneous tissue disorders", "Congenital, familial and genetic disorders", "Infections and infestations", "Respiratory, thoracic and mediastinal disorders", "Psychiatric disorders", "Renal and urinary disorders", "Pregnancy, puerperium and perinatal conditions", "Ear and labyrinth disorders", "Cardiac disorders", "Nervous system disorders", "Injury, poisoning and procedural complications"], 'task': 'classification'},
}

def get_git_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "N/A"

def run_single_target_experiment(base_config, dataset_name, target_name, run_id, n_repeats):
    print(f"\n--- [Run {run_id}/{n_repeats}] Training for {dataset_name} - {target_name} ---")
    config = base_config.copy()
    ds_config = DATASET_CONFIGS[dataset_name]

    config['task_name'] = f"{dataset_name}_{target_name}_run{run_id}"
    
    wrapper_args = {
        'batch_size': config['batch_size'],
        'num_workers': config['dataset']['num_workers'],
        'valid_size': config['dataset']['valid_size'],
        'test_size': config['dataset']['test_size'],
        'data_path': ds_config['data_path'],
        'target': target_name,
        'task': ds_config['task'],
        'splitting': config['dataset']['splitting']
    }
    dataset_wrapper = HybridDatasetWrapper(**wrapper_args)
    
    config['dataset'].update({
        'data_path': ds_config['data_path'],
        'target': target_name,
        'task': ds_config['task'],
        'metric': 'roc_auc',
        'n_class': 1
    })
    
    trainer = StandaloneTrainer(dataset_wrapper, config)
    score = trainer.train()
    
    print(f"--- Final Test Score for '{target_name}': {score:.4f} ---")
    return score

# 파일: attention_ratio_finetune.py

# ... (파일 상단의 모든 클래스와 함수 정의는 그대로 유지) ...

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Cross-Attention Ratio (alpha/beta) model benchmark.")
    parser.add_argument('--dataset', type=str, default='all', help="Name of the dataset to run. Use 'all' for all datasets. Choices: " + ", ".join(list(DATASET_CONFIGS.keys())))
    parser.add_argument('--repeats', type=int, default=3, help="Number of times to repeat the experiments for statistical validation.")
    parser.add_argument('--config', type=str, default='config_cross_attention_ratio.yaml', help="Path to the configuration YAML file for ratio experiments.")
    args = parser.parse_args()
    
    try:
        with open(args.config, "r", encoding="utf-8") as f:
            base_config = yaml.load(f, Loader=yaml.FullLoader)
    except FileNotFoundError:
        print(f"ERROR: Config file not found at {args.config}")
        exit(1)
        
    base_config['git_hash'] = get_git_hash()

    if args.dataset.lower() == 'all':
        datasets_to_run = list(DATASET_CONFIGS.keys())
        print("--- Running experiments for ALL datasets ---")
    elif args.dataset in DATASET_CONFIGS:
        datasets_to_run = [args.dataset]
        print(f"--- Running experiment for dataset: {args.dataset} ---")
    else:
        print(f"Error: Dataset '{args.dataset}' not found. Please choose from 'all' or {list(DATASET_CONFIGS.keys())}")
        exit(1)

    os.makedirs('results', exist_ok=True)
    results_log = []

    for i in range(args.repeats):
        run_number = i + 1
        print(f"\n\n{'='*25} STARTING RUN {run_number}/{args.repeats} {'='*25}")
        
        for name in datasets_to_run:
            target_list = DATASET_CONFIGS[name]['target']
            if not isinstance(target_list, list):
                target_list = [target_list]
                
            for target in target_list:
                score = run_single_target_experiment(base_config, name, target, run_number, args.repeats)
                results_log.append({
                    'dataset': name,
                    'target': target,
                    'run_id': run_number,
                    'score': score
                })
    
    print(f"\n\n{'='*30} FINAL RESULTS (after {args.repeats} runs) {'='*30}")
    results_df = pd.DataFrame(results_log)
    
    if not results_df.empty:
        # 1. 타겟별 평균/표준편차 계산 (기존과 동일)
        summary_per_target = results_df.groupby(['dataset', 'target'])['score'].agg(['mean', 'std']).reset_index()
        summary_per_target['std'] = summary_per_target['std'].fillna(0)
        summary_per_target['performance'] = summary_per_target.apply(
            lambda row: f"{row['mean']:.4f} ± {row['std']:.4f}", axis=1)

        # 2. 데이터셋별 평균 계산 (기존과 동일)
        summary_per_dataset = results_df.groupby('dataset')['score'].agg(['mean', 'std']).reset_index()
        summary_per_dataset['std'] = summary_per_dataset['std'].fillna(0)
        summary_per_dataset['performance'] = summary_per_dataset.apply(
            lambda row: f"{row['mean']:.4f} ± {row['std']:.4f}", axis=1)
        
        # --- [핵심 수정] 3. 전체 종합 평균 점수 계산 ---
        # 멀티-타겟 데이터셋(Tox21, SIDER 등)의 경우, 각 타겟의 평균 점수를 먼저 구한 뒤,
        # 이 데이터셋별 평균 점수들의 최종 평균을 계산합니다.
        # 이렇게 해야 각 데이터셋이 동등한 가중치를 갖게 됩니다.
        overall_average_score = summary_per_dataset['mean'].mean()
        
        # --- 터미널에 최종 결과 출력 ---
        print("\n--- Summary per Target ---")
        print(summary_per_target[['dataset', 'target', 'performance']].to_string(index=False))

        print("\n--- Overall Summary per Dataset ---")
        print(summary_per_dataset[['dataset', 'performance']].to_string(index=False))

        # --- [핵심 수정] 전체 종합 평균 점수 출력 ---
        print("\n" + "="*20 + " FINAL OVERALL AVERAGE SCORE " + "="*20)
        print(f"The overall average score across all datasets is: {overall_average_score:.4f}")
        print("="*65)


        # --- CSV 파일로 저장 (기존과 동일) ---
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        git_hash = base_config.get('git_hash', 'N/A')
        filename_prefix = f"{args.dataset}_ratio" if args.dataset.lower() != 'all' else 'all_datasets_ratio'
        
        detailed_save_path = f'results/{filename_prefix}_detailed_{git_hash}_{timestamp}.csv'
        results_df.to_csv(detailed_save_path, index=False)
        print(f"\nDetailed run-by-run results saved to: {detailed_save_path}")
        
        summary_target_save_path = f'results/{filename_prefix}_summary_per_target_{git_hash}_{timestamp}.csv'
        summary_per_target.to_csv(summary_target_save_path, index=False)
        print(f"Summary results per target saved to: {summary_target_save_path}")

        # 데이터셋별 요약 결과도 별도로 저장
        summary_dataset_save_path = f'results/{filename_prefix}_summary_per_dataset_{git_hash}_{timestamp}.csv'
        summary_per_dataset.to_csv(summary_dataset_save_path, index=False)
        print(f"Overall summary results per dataset saved to: {summary_dataset_save_path}")

        config_save_path = f'results/config_{filename_prefix}_{git_hash}_{timestamp}.yaml'
        with open(config_save_path, 'w') as f:
            yaml.dump(base_config, f, default_flow_style=False)
        print(f"Config file used for this run saved to: {config_save_path}")
    else:
        print("No results were generated.")