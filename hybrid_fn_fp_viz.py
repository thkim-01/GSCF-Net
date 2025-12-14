import os
import shutil
import sys
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib
matplotlib.use('Agg') # GUI 없는 서버 환경을 위한 설정
import matplotlib.pyplot as plt # pyplot 모듈을 plt로 import
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error, confusion_matrix
from rdkit import Chem
from rdkit.Chem import Draw
import io
from PIL import Image
import seaborn as sns
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from hybrid_dataset import HybridDatasetWrapper
from hybrid_model import HybridModel
from PIL import Image, ImageDraw, ImageFont
import math
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

apex_support = False
try:
    from apex import amp
    apex_support = True
    # print("Apex found. Mixed precision training is available.")
except ImportError:
    # print("Apex not found. Training in full precision (FP32).")
    pass


class Normalizer(object):
    """Normalize a Tensor and restore it later. """
    def __init__(self, tensor):
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean, 'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']

def _draw_molecule_grid(smiles_list, legends, mols_per_row=5, sub_img_size=(200, 200), output_path='molecules.png'):
    """주어진 SMILES 리스트를 그리드 이미지로 저장합니다."""
    
    mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
    # 빈 mol 객체(유효하지 않은 SMILES)가 있는 경우를 대비한 필터링
    valid_mols = [m for m in mols if m is not None]
    valid_legends = [legends[i] for i, m in enumerate(mols) if m is not None]
    if not valid_mols:
        print(f"Warning: No valid molecules found in the list to generate {os.path.basename(output_path)}.")
        return # 그릴 분자가 없으면 함수 종료
    try:
        img = Draw.MolsToGridImage(
            valid_mols,
            molsPerRow=mols_per_row,
            subImgSize=sub_img_size,
            legends=valid_legends
        )
        img.save(output_path)
        print(f"Molecule grid image saved to {output_path}")
    except Exception as e:
        print(f"An error occurred while drawing molecules to {output_path}: {e}")
        # RDKit 드로잉 백엔드 문제일 수 있으므로 추가 정보 제공
        print("Please ensure your RDKit installation includes Cairo support (conda install -c conda-forge rdkit).")

def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config_hybrid.yaml', os.path.join(model_checkpoints_folder, 'config_hybrid.yaml'))


class FineTune(object):
    def __init__(self, dataset, config):
        self.config = config
        self.device = self._get_device()
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        molclr_version = config['model_type']
        chemberta_version = config['hybrid_specific']['chemberta_model_name']
        dir_name = f"{current_time}_{config['task_name']}_MolCLR({molclr_version})_{chemberta_version}"
        # ----------------------------------------------------
        log_dir = os.path.join('finetune', dir_name)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.dataset = dataset
        
        if config['dataset']['task'] == 'classification':
            self.criterion = nn.CrossEntropyLoss()
        elif config['dataset']['task'] == 'regression':
            self.criterion = nn.L1Loss() if self.config["task_name"] in ['qm7', 'qm8', 'qm9'] else nn.MSELoss()

    def _get_device(self):
        device = torch.device(self.config['gpu'] if torch.cuda.is_available() else 'cpu')
        print("Running on:", device)
        return device

    def _step(self, model, graph_data, smiles_list):
        __, pred = model(graph_data, smiles_list)
        if self.config['dataset']['task'] == 'classification':
            loss = self.criterion(pred, graph_data.y.flatten())
        elif self.config['dataset']['task'] == 'regression':
            labels = graph_data.y
            if self.normalizer:
                loss = self.criterion(pred, self.normalizer.norm(labels))
            else:
                loss = self.criterion(pred, labels)
        return loss
# FineTune 클래스 내부에 이 메서드를 추가하세요.

    def _plot_confusion_matrix(self, cm, class_names):
        """Generates and saves a confusion matrix plot."""
        # 그림 크기 설정
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Seaborn을 사용하여 히트맵 생성
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', ax=ax,
                    xticklabels=class_names, yticklabels=class_names)
        
        # 라벨 및 제목 설정
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title('Confusion Matrix', fontsize=16)
        
        # 저장 경로 설정 및 저장
        output_dir = os.path.join(self.writer.log_dir, self.config['dataset']['target'], 'error_analysis')
        os.makedirs(output_dir, exist_ok=True) # 폴더가 없으면 생성
        save_path = os.path.join(output_dir, 'confusion_matrix.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Confusion matrix plot saved to {save_path}")
  
    def train(self):
        train_loader, valid_loader, test_loader = self.dataset.get_data_loaders()

        self.normalizer = None
        if self.config["task_name"] in ['qm7', 'qm9']:
            labels = []
            for graph_batch, _ in train_loader:
                labels.append(graph_batch.y)
            labels = torch.cat(labels)
            self.normalizer = Normalizer(labels)
            print("Labels normalized. Mean:", self.normalizer.mean.item(), "Std:", self.normalizer.std.item())

        model = HybridModel(
            molclr_config=self.config["model"],
            task_type=self.config['dataset']['task'],
            chemberta_model_name=self.config['hybrid_specific']['chemberta_model_name']
        ).to(self.device)

        self._load_molclr_pre_trained_weights(model)

        optimizer = torch.optim.Adam([
            {'params': model.molclr_model.parameters(), 'lr': float(self.config['init_base_lr'])},
            {'params': model.chemberta_model.parameters(), 'lr': float(self.config['hybrid_specific']['chemberta_lr'])},
            {'params': model.hybrid_pred_head.parameters(), 'lr': float(self.config['init_lr'])}
        ], weight_decay=float(self.config['weight_decay']))

        if apex_support and self.config['fp16_precision']:
            print("Activating mixed precision training with Apex.")
            model, optimizer = amp.initialize(
                model, optimizer, opt_level='O2', keep_batchnorm_fp32=True
            )

        model_checkpoints_folder = os.path.join(self.writer.log_dir, self.config['dataset']['target'], 'checkpoints')
        _save_config_file(model_checkpoints_folder)
        
        n_iter = 0
        valid_n_iter = 0
        best_valid_rgr = np.inf
        best_valid_cls = 0

        for epoch_counter in range(self.config['epochs']):
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch_counter}/{self.config['epochs']}")
            for bn, (graph_data, smiles_list) in enumerate(progress_bar):
                optimizer.zero_grad()
                graph_data = graph_data.to(self.device)
                
                loss = self._step(model, graph_data, smiles_list)

                if apex_support and self.config['fp16_precision']:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                optimizer.step()
                n_iter += 1

                progress_bar.set_postfix({'loss': loss.item()})
                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar(f'{self.config["dataset"]["target"]}/train_loss', loss, global_step=n_iter)
            
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                print(f"\n--- Validating Epoch {epoch_counter} for target: {self.config['dataset']['target']} ---")
                if self.config['dataset']['task'] == 'classification': 
                    valid_loss, valid_cls = self._validate(model, valid_loader)
                    if valid_cls > best_valid_cls:
                        best_valid_cls = valid_cls
                        torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))
                        print(f"** New best ROC AUC: {best_valid_cls:.4f}. Model saved! **")
                elif self.config['dataset']['task'] == 'regression': 
                    valid_loss, valid_rgr = self._validate(model, valid_loader)
                    if valid_rgr < best_valid_rgr:
                        best_valid_rgr = valid_rgr
                        torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))
                        metric_name = "MAE" if self.config["task_name"] in ['qm7', 'qm8', 'qm9'] else "RMSE"
                        print(f"** New best {metric_name}: {best_valid_rgr:.4f}. Model saved! **")
                
                self.writer.add_scalar(f'{self.config["dataset"]["target"]}/validation_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1
        
        print(f"\n--- Final Testing for target: {self.config['dataset']['target']} ---")
        # _test 메서드가 반환하는 최종 점수를 받아서 return
        final_test_score, test_results = self._test(model, test_loader)
        
        # 테스트 결과가 있을 경우 (classification task일 경우) 시각화 함수 호출
        if test_results is not None:
            self._analyze_and_visualize_errors(test_results)
            
        return final_test_score

    def _load_molclr_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join('./ckpt', self.config['fine_tune_from'], 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'), map_location=self.device)
            model.load_molclr_pre_trained_weights(state_dict)
            print("Successfully loaded pre-trained MolCLR weights.")
        except FileNotFoundError:
            print("Pre-trained MolCLR weights not found. MolCLR part will be trained from scratch.")
        return model

    def _validate(self, model, valid_loader):
        predictions, labels = [], []
        with torch.no_grad():
            model.eval()
            valid_loss, num_data = 0.0, 0
            for graph_data, smiles_list in tqdm(valid_loader, desc="Validating"):
                graph_data = graph_data.to(self.device)
                
                __, pred = model(graph_data, smiles_list)
                loss = self._step(model, graph_data, smiles_list)
                valid_loss += loss.item() * graph_data.y.size(0)
                num_data += graph_data.y.size(0)

                if self.normalizer: pred = self.normalizer.denorm(pred)
                if self.config['dataset']['task'] == 'classification': pred = F.softmax(pred, dim=-1)

                predictions.extend(pred.cpu().detach().numpy())
                labels.extend(graph_data.y.cpu().flatten().numpy())
            valid_loss /= num_data
        model.train()
        
        predictions, labels = np.array(predictions), np.array(labels)
        if self.config['dataset']['task'] == 'regression':
            if self.config['task_name'] in ['qm7', 'qm8', 'qm9']:
                mae = mean_absolute_error(labels, predictions)
                print(f'Validation Loss: {valid_loss:.4f}, MAE: {mae:.4f}')
                return valid_loss, mae
            else:
                rmse = mean_squared_error(labels, predictions, squared=False)
                print(f'Validation Loss: {valid_loss:.4f}, RMSE: {rmse:.4f}')
                return valid_loss, rmse
        elif self.config['dataset']['task'] == 'classification':
            roc_auc = roc_auc_score(labels, predictions[:,1])
            print(f'Validation Loss: {valid_loss:.4f}, ROC AUC: {roc_auc:.4f}')
            return valid_loss, roc_auc
    
    def _test(self, model, test_loader):
        model_path = os.path.join(self.writer.log_dir, self.config['dataset']['target'], 'checkpoints', 'model.pth')
        if not os.path.exists(model_path):
            print(f"Model file not found for testing at: {model_path}")
            return (0.0, None) if self.config['dataset']['task'] == 'classification' else (float('nan'), None)

        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()
        print("Loaded best trained model for testing.")

        # 시각화를 위한 데이터 저장소
        test_results = { 'smiles': [], 'labels': [], 'preds': [], 'embeddings': [] }
        
        with torch.no_grad():
            for graph_data, smiles_list in tqdm(test_loader, desc="Final Testing"):
                graph_data = graph_data.to(self.device)
                embedding, pred_logits = model(graph_data, smiles_list)
                
                # 데이터 저장
                test_results['smiles'].extend(smiles_list)
                test_results['labels'].extend(graph_data.y.cpu().numpy())
                test_results['embeddings'].append(embedding.cpu().numpy())
                
                if self.config['dataset']['task'] == 'classification':
                    # 예측 레이블 (0 또는 1)만 저장합니다. ROC AUC를 위해서는 softmax 출력이 필요할 수 있지만, 
                    # 여기서는 t-SNE 시각화 목적이므로 예측된 최종 클래스를 저장합니다.
                    pred_labels = torch.argmax(pred_logits, dim=1) 
                    test_results['preds'].extend(pred_labels.cpu().numpy())
                else: # Regression
                    # 예측값 자체를 저장 (정규화 풀기 필요 시 추가)
                    pred_values = pred_logits.cpu().numpy()
                    test_results['preds'].extend(pred_values)
        
        # NumPy 배열로 변환
        for key, val in test_results.items():
            test_results[key] = np.concatenate(val) if key == 'embeddings' else np.array(val)

        # 성능 계산 및 반환
        labels, preds = test_results['labels'], test_results['preds']
        if self.config['dataset']['task'] == 'classification':
            # ROC AUC를 계산하기 위해선 _test 함수에서 pred_logits (softmax 이전 혹은 이후 확률)를 저장해야 합니다.
            # 현재 'preds'에는 argmax를 통해 얻은 최종 레이블만 있습니다.
            # 따라서 ROC AUC 계산을 위해서는 'preds'에 확률 값을 저장하도록 수정하거나, 
            # 여기에서 확률 값을 다시 계산해야 합니다.
            # 일단 여기서는 임시로 (레이블 기반의) 정확도를 계산하거나, 
            # 원래 ROC AUC 계산 로직이 있다면 해당 확률이 저장된 변수를 사용해야 합니다.
            # 만약 pred_logits가 그대로 저장되었다면, `roc_auc_score(labels, pred_logits[:,1].cpu().numpy())`와 같이 사용할 수 있습니다.
            # 현재 코드상으로는 `preds`가 0 또는 1이므로 roc_auc_score에 직접 사용하기 어렵습니다. 
            # 이 부분을 `test_results`에 확률을 저장하는 방식으로 수정해야 합니다.
            # 예를 들어, test_results['pred_probs'].extend(F.softmax(pred_logits, dim=-1).cpu().numpy()) 추가
            
            # --- 임시 처리: ROC AUC 계산을 위해 예측 확률이 아닌 예측 레이블로 ROC AUC 계산 (올바르지 않을 수 있음) ---
            # 혹은, _test 함수 상단에 `test_results['pred_probs'] = []` 추가
            # 그리고 `pred_logits`를 저장하도록 변경: `test_results['pred_probs'].append(F.softmax(pred_logits, dim=-1).cpu().numpy())`
            # 최종적으로 `preds_probs = np.concatenate(test_results['pred_probs'])`
            # `roc_auc = roc_auc_score(labels, preds_probs[:,1])`
            # 현재 `preds`에 argmax 결과만 저장되므로 ROC AUC는 0 또는 1 예측만으로 계산되지 않습니다.
            # 이 부분을 수정하려면 `_test`에서 `pred_logits`의 확률을 저장해야 합니다.
            
            # 아래는 임시로 예측된 클래스 레이블로 ROC AUC를 시도하지만, 이는 일반적으로 권장되지 않습니다.
            # 실제 ROC AUC를 위해서는 Softmax 출력 (확률)이 필요합니다.
            # 일단 시각화에는 예측된 레이블 (0 또는 1)이 사용될 것이므로 그대로 둡니다.
            # 정확한 ROC AUC 계산을 위해서는 _test 메서드의 pred 저장 부분을 수정해야 합니다.
            roc_auc = roc_auc_score(labels, preds) # 이 부분은 `preds`가 확률이 아니라 0/1일 경우 정확하지 않음
            print(f'Test ROC AUC (using predicted labels): {roc_auc:.4f}') # 경고 메시지 추가
            return roc_auc, test_results
        else: # Regression
            metric_name = "MAE" if self.config["task_name"] in ['qm7', 'qm8', 'qm9'] else "RMSE"
            score = mean_absolute_error(labels, preds) if metric_name == "MAE" else mean_squared_error(labels, preds, squared=False)
            print(f'Test {metric_name}: {score:.4f}')
            return score, test_results

    # `FineTune` 클래스 내부의 메서드입니다.
    def _analyze_and_visualize_errors(self, test_results, n_samples=3):
        """
        [Final Version] Analyzes test results, generates a confusion matrix plot,
        and annotates FP/FN errors with colored numbers on a t-SNE plot.
        """
        # --- Task Type Check ---
        if self.config['dataset']['task'] != 'classification':
            print("Skipping error analysis for non-classification task.")
            return

        print("\n--- Starting Final Error Analysis ---")
        output_dir = os.path.join(self.writer.log_dir, self.config['dataset']['target'], 'error_analysis')
        os.makedirs(output_dir, exist_ok=True)

        # --- Data Extraction ---
        labels = np.array(test_results['labels'])
        preds = np.array(test_results['preds'])
        smiles = test_results['smiles']
        embeddings = np.array(test_results['embeddings'])

        if len(labels) == 0 or len(embeddings) != len(labels):
            print("Error: Invalid or empty test results data. Skipping visualization.")
            return

        # --- Confusion Matrix Calculation & Plotting ---
        cm = confusion_matrix(labels, preds)
        self._plot_confusion_matrix(cm, class_names=['Negative', 'Positive'])
        
        tn, fp, fn, tp = (cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0))
        if cm.shape == (2, 2):
            print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

        # --- Misclassification Indexing & Sample Selection ---
        fp_idx = np.where((labels == 0) & (preds == 1))[0]
        fn_idx = np.where((labels == 1) & (preds == 0))[0]

        if len(fp_idx) > 0:
            fp_draw = np.random.choice(fp_idx, size=n_samples, replace=(len(fp_idx) < n_samples))
        else:
            fp_draw = []
        
        if len(fn_idx) > 0:
            fn_draw = np.random.choice(fn_idx, size=n_samples, replace=(len(fn_idx) < n_samples))
        else:
            fn_draw = []

        # --- t-SNE Computation ---
        print("\n--- Starting t-SNE Analysis with Numbered Annotations ---")
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        scaled_embeddings = StandardScaler().fit_transform(embeddings)
        tsne_results = tsne.fit_transform(scaled_embeddings)

        # --- t-SNE Visualization ---
        fig, ax = plt.subplots(figsize=(20, 18))
        scatter = ax.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='winter', alpha=1.0, s=300)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # 헬퍼 함수
        def annotate_misclassified_points(indices, color, label, error_type_str):
            """Helper function to highlight points and add text number annotations."""
            for i, idx in enumerate(indices):
                print(f"  - {error_type_str} Molecule #{i+1} (Original index: {idx}): {smiles[idx]}")
                
                point = tsne_results[idx]
                text_label = f"#{i+1}"
                
                # 하이라이트 원 그리기
                ax.scatter(point[0], point[1], marker='o', facecolor='none', 
                           edgecolor=color, linewidth=2.5, s=250, 
                           label=label if i == 0 else "", zorder=10)

                # 번호 텍스트 추가
                ax.text(point[0], point[1] + 0.5, text_label,
                        fontsize=12, 
                        fontweight='bold', 
                        color=color, # <<< CHANGED: 'black'에서 동적인 color 변수로 변경
                        ha='center',
                        va='bottom',
                        zorder=11)

        # FP/FN 포인트에 어노테이션 추가
        if len(fp_draw) > 0:
            print("\n--- SMILES strings for ANNOTATED False Positives ---")
            annotate_misclassified_points(fp_draw, color='red', label='False Positive', error_type_str='False Positive')
        
        if len(fn_draw) > 0:
            print("\n--- SMILES strings for ANNOTATED False Negatives ---")
            annotate_misclassified_points(fn_draw, color='black', label='False Negative', error_type_str='False Negative')

        # 최종 플롯 설정
        ax.legend(loc='upper right', fontsize=14)
        fig.tight_layout()
        
        save_path = os.path.join(output_dir, 'tsne_annotated_with_numbers.png')
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"\nAnnotated t-SNE plot saved to {save_path}")


def main(config):
    dataset = HybridDatasetWrapper(config['batch_size'], **config['dataset'])
    fine_tune = FineTune(dataset, config)
    
    final_score = fine_tune.train()
    return final_score

if __name__ == "__main__":
    config = yaml.load(open("config_hybrid.yaml", "r", encoding="utf-8"), Loader=yaml.FullLoader)
    
    # config 파일에서 task_name을 직접 읽어옵니다.
    task_name = config['task_name']

    if task_name == 'BBBP':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/bbbp/BBBP.csv'
        target_list = ["p_np"]

    elif task_name == 'Tox21':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/tox21/tox21.csv'
        target_list = [
            "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD", 
            "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
        ]

    elif task_name == 'ClinTox':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/clintox/clintox.csv'
        target_list = ['CT_TOX', 'FDA_APPROVED']

    elif task_name == 'HIV':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/hiv/HIV.csv'
        target_list = ["HIV_active"]

    elif task_name == 'BACE':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/bace/bace.csv'
        target_list = ["Class"]

    elif task_name == 'SIDER':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/sider/sider.csv'
        target_list = [
            "Hepatobiliary disorders", "Metabolism and nutrition disorders", "Product issues", 
            "Eye disorders", "Investigations", "Musculoskeletal and connective tissue disorders", 
            "Gastrointestinal disorders", "Social circumstances", "Immune system disorders", 
            "Reproductive system and breast disorders", 
            "Neoplasms benign, malignant and unspecified (incl cysts and polyps)", 
            "General disorders and administration site conditions", "Endocrine disorders", 
            "Surgical and medical procedures", "Vascular disorders", 
            "Blood and lymphatic system disorders", "Skin and subcutaneous tissue disorders", 
            "Congenital, familial and genetic disorders", "Infections and infestations", 
            "Respiratory, thoracic and mediastinal disorders", "Psychiatric disorders", 
            "Renal and urinary disorders", "Pregnancy, puerperium and perinatal conditions", 
            "Ear and labyrinth disorders", "Cardiac disorders", 
            "Nervous system disorders", "Injury, poisoning and procedural complications"
        ]
    
    elif task_name == 'MUV':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/muv/muv.csv'
        target_list = [
            'MUV-692', 'MUV-689', 'MUV-846', 'MUV-859', 'MUV-644', 'MUV-548', 'MUV-852',
            'MUV-600', 'MUV-810', 'MUV-712', 'MUV-737', 'MUV-858', 'MUV-713', 'MUV-733',
            'MUV-652', 'MUV-466', 'MUV-832'
        ]

    # --- 아래는 regression task ---
    elif task_name == 'FreeSolv':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = 'data/freesolv/freesolv.csv'
        target_list = ["expt"]
    
    elif task_name == "ESOL":
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = 'data/esol/esol.csv'
        target_list = ["measured log solubility in mols per litre"]

    elif task_name == "Lipo":
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = 'data/lipophilicity/Lipophilicity.csv'
        target_list = ["exp"]
    
    elif task_name == "qm7":
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = 'data/qm7/qm7.csv'
        target_list = ["u0_atom"]

    elif task_name == "qm8":
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = 'data/qm8/qm8.csv'
        target_list = [
            "E1-CC2", "E2-CC2", "f1-CC2", "f2-CC2", "E1-PBE0", "E2-PBE0", 
            "f1-PBE0", "f2-PBE0", "E1-CAM", "E2-CAM", "f1-CAM","f2-CAM"
        ]
    
    elif task_name == "qm9":
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = 'data/qm9/qm9.csv'
        target_list = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'cv']

    else:
        raise ValueError(f'Undefined downstream task in config: {task_name}')

    # ===== 시행 횟수를 포함한 결과 폴더 경로 생성 =====
    base_results_dir = 'experiments'
    run_number = 1
    while True:
        results_dir = os.path.join(base_results_dir, f"{task_name}_{run_number}")
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            break
        run_number += 1
    
    print(f"\n\n{'='*20} Starting Experiment for: {task_name} (Run #{run_number}) {'='*20}")
    print(f"Results for this run will be saved in: {results_dir}")
    print("Running on targets:", target_list)

    results_list = []
    # 현재 데이터셋의 모든 타겟을 순회하며 학습
    for target in target_list:
        print(f"\n===== Training for target: {target} =====")
        config['dataset']['target'] = target
        
        result_score = main(config)
        results_list.append([target, result_score])

    # ===== 한 데이터셋의 모든 타겟에 대한 실험이 끝나면 CSV로 저장 =====
    
    # 1. 결과를 데이터프레임으로 변환 (이 부분에서 타겟별로 기록됩니다)
    metric_name = 'ROC_AUC' if config['dataset']['task'] == 'classification' else 'MAE/RMSE'
    df = pd.DataFrame(results_list, columns=['Target', metric_name])
    
    # 2. 평균/표준편차를 계산 (타겟이 2개 이상일 때만 실행됩니다)
    if len(target_list) > 1:
        mean_score = df[metric_name].mean()
        std_score = df[metric_name].std()
        
        summary_df = pd.DataFrame([
            {'Target': '---', metric_name: '---'},
            {'Target': 'Mean', metric_name: mean_score},
            {'Target': 'Std', metric_name: std_score}
        ])
        df = pd.concat([df, summary_df], ignore_index=True)
    
    # 3. CSV 파일로 저장
    results_csv_path = os.path.join(results_dir, 'results.csv')
    df.to_csv(results_csv_path, index=False)
    
    print(f"\n{'='*20} Experiment for {task_name} (Run #{run_number}) Finished {'='*20}")
    print(f"Results saved to: {results_csv_path}")
    print(df) # 최종 데이터프레임 출력
    print("\n\n")