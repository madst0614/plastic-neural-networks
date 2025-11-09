"""
Experimental Evidence for Plastic Neural Networks
실험적 증거 찾기 - PNN의 뇌과학적 타당성 검증

이 스크립트는 세 가지 실험적 접근법을 시뮬레이션합니다:
1. MEG (High-temporal resolution) - Gamma cycle 내 활동 패턴 분석
2. Optogenetics - 특정 뉴런/레이어 억제 실험
3. Brain Activity Modeling - 모델 예측 vs 실제 뇌 활동 비교

Usage:
    python scripts/experimental_evidence.py --checkpoint checkpoints/best_model.pt
    python scripts/experimental_evidence.py --checkpoint checkpoints/best_model.pt --experiment meg
    python scripts/experimental_evidence.py --checkpoint checkpoints/best_model.pt --experiment optogenetics
    python scripts/experimental_evidence.py --checkpoint checkpoints/best_model.pt --experiment modeling
"""

# Add parent directory to path for imports (makes it work without installation)
import sys
from pathlib import Path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional

from pnn.models.pnn import create_pnn_model
from pnn.utils.training import load_checkpoint
from transformers import BertTokenizer
from datasets import load_dataset


# Define Config class at module level for checkpoint compatibility
class Config:
    """Dummy Config class for checkpoint loading compatibility"""
    pass


class MEGSimulator:
    """
    MEG (Magnetoencephalography) 시뮬레이션

    Gamma cycle (millisecond 해상도) 내에서 활동 패턴 분석:
    - 초반: Delta generation (높은 activity)
    - 중반: Gate computation (특정 패턴)
    - 후반: Update (다른 패턴)
    """

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.eval()

        # Hook을 사용해서 중간 활동 기록
        self.activations = {}
        self.register_hooks()

    def register_hooks(self):
        """중간 레이어 활동을 기록하기 위한 hook 등록"""
        def get_activation(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    output = output[0]
                self.activations[name] = output.detach()
            return hook

        # DeltaRefiner의 주요 컴포넌트에 hook 등록
        self.model.delta_refiner.attention.register_forward_hook(
            get_activation('attention_output')
        )
        self.model.delta_refiner.ffn.register_forward_hook(
            get_activation('ffn_output')
        )
        self.model.delta_refiner.gate.register_forward_hook(
            get_activation('gate_output')
        )

    def simulate_gamma_cycle(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Dict[str, List[torch.Tensor]]:
        """
        Gamma cycle 시뮬레이션

        각 refinement step을 gamma cycle로 간주하고,
        cycle 내에서의 활동 패턴을 분석

        Returns:
            step_activities: 각 step별 활동 패턴
        """
        step_activities = {
            'delta_generation': [],
            'gate_computation': [],
            'hidden_update': [],
            'activity_magnitude': []
        }

        with torch.no_grad():
            # Embedding
            batch_size, seq_len = input_ids.shape
            token_embeds = self.model.token_embeddings(input_ids)
            position_ids = torch.arange(seq_len, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            position_embeds = self.model.position_embeddings(position_ids)

            hidden = token_embeds + position_embeds
            hidden = self.model.embedding_layer_norm(hidden)
            hidden = self.model.embedding_dropout(hidden)

            attn_mask = (attention_mask == 0) if attention_mask is not None else None

            # 각 refinement step을 gamma cycle로 분석
            for step in range(self.model.num_steps):
                self.activations.clear()

                # Delta 생성
                delta = self.model.delta_refiner(hidden, attn_mask)

                # 활동 패턴 기록
                # 1. Delta generation (초반 - 높은 activity)
                if 'ffn_output' in self.activations:
                    delta_activity = self.activations['ffn_output']
                    step_activities['delta_generation'].append(
                        delta_activity.abs().mean().cpu()
                    )

                # 2. Gate computation (중반 - 특정 패턴)
                if 'gate_output' in self.activations:
                    gate_activity = self.activations['gate_output']
                    step_activities['gate_computation'].append(
                        gate_activity.mean().cpu()
                    )

                # 3. Hidden update (후반)
                hidden_before = hidden.clone()
                hidden = hidden + delta
                update_magnitude = (hidden - hidden_before).abs().mean()
                step_activities['hidden_update'].append(update_magnitude.cpu())

                # 전체 activity magnitude
                total_activity = delta.abs().mean()
                step_activities['activity_magnitude'].append(total_activity.cpu())

        return step_activities

    def analyze_temporal_patterns(
        self,
        dataloader,
        num_batches: int = 10
    ) -> Dict:
        """
        여러 배치에 걸쳐 temporal pattern 분석

        Returns:
            analysis: 통계 및 패턴 분석 결과
        """
        all_activities = {
            'delta_generation': [],
            'gate_computation': [],
            'hidden_update': [],
            'activity_magnitude': []
        }

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="MEG Analysis", total=num_batches)):
            if batch_idx >= num_batches:
                break

            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)

            activities = self.simulate_gamma_cycle(input_ids, attention_mask)

            for key in all_activities:
                all_activities[key].extend(activities[key])

        # 통계 분석
        analysis = {}
        for key, values in all_activities.items():
            values = torch.tensor(values).numpy()
            analysis[key] = {
                'mean': float(values.mean()),
                'std': float(values.std()),
                'values': values.tolist()
            }

        return analysis


class OptogeneticsSimulator:
    """
    Optogenetics 시뮬레이션

    특정 뉴런/레이어를 억제하여 행동 변화 관찰:
    - Attention 억제 → Delta 생성 막힘?
    - Gate 억제 → Update 막힘?
    - FFN 억제 → 정보 처리 변화?
    """

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.eval()

    def suppress_component(
        self,
        component: str,
        suppression_rate: float = 1.0
    ):
        """
        특정 컴포넌트 억제

        Args:
            component: 'attention', 'gate', 'ffn' 중 하나
            suppression_rate: 억제 비율 (0.0 = 억제 없음, 1.0 = 완전 억제)
        """
        class SuppressionHook:
            def __init__(self, rate):
                self.rate = rate

            def __call__(self, module, input, output):
                if isinstance(output, tuple):
                    output = list(output)
                    output[0] = output[0] * (1.0 - self.rate)
                    return tuple(output)
                else:
                    return output * (1.0 - self.rate)

        hook = SuppressionHook(suppression_rate)

        if component == 'attention':
            handle = self.model.delta_refiner.attention.register_forward_hook(hook)
        elif component == 'gate':
            handle = self.model.delta_refiner.gate.register_forward_hook(hook)
        elif component == 'ffn':
            handle = self.model.delta_refiner.ffn.register_forward_hook(hook)
        else:
            raise ValueError(f"Unknown component: {component}")

        return handle

    def measure_behavior(
        self,
        dataloader,
        labels_key: str = 'labels',
        num_batches: int = 10
    ) -> Dict:
        """
        모델 행동 측정 (정확도, 손실 등)

        Returns:
            metrics: 성능 지표
        """
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Measuring", total=num_batches)):
                if batch_idx >= num_batches:
                    break

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch[labels_key].to(self.device)

                hidden = self.model(input_ids, attention_mask)
                loss, logits = self.model.get_mlm_loss(hidden, labels)

                total_loss += loss.item()

                preds = logits.argmax(dim=-1)
                mask = (labels != -100)
                correct = (preds == labels) & mask
                total_correct += correct.sum().item()
                total_tokens += mask.sum().item()

        return {
            'loss': total_loss / num_batches,
            'accuracy': total_correct / total_tokens if total_tokens > 0 else 0,
            'total_tokens': total_tokens
        }

    def run_suppression_experiment(
        self,
        dataloader,
        num_batches: int = 10
    ) -> Dict:
        """
        여러 컴포넌트에 대한 억제 실험

        Returns:
            results: 각 조건별 실험 결과
        """
        results = {}

        # 1. Baseline (억제 없음)
        print("\n📊 Baseline (No suppression)...")
        results['baseline'] = self.measure_behavior(dataloader, num_batches=num_batches)

        # 2. Attention 억제
        components = ['attention', 'gate', 'ffn']
        suppression_rates = [0.25, 0.5, 0.75, 1.0]

        for component in components:
            for rate in suppression_rates:
                print(f"\n🔬 Suppressing {component} at {rate*100:.0f}%...")

                # 억제 적용
                handle = self.suppress_component(component, rate)

                # 행동 측정
                metrics = self.measure_behavior(dataloader, num_batches=num_batches)

                # 결과 저장
                key = f"{component}_suppressed_{int(rate*100)}"
                results[key] = metrics

                # 억제 해제
                handle.remove()

        return results


class BrainActivityPredictor:
    """
    Brain Activity Modeling

    모델의 활동 패턴을 예측하고 실제 뇌 활동과 비교
    (실제 fMRI/MEG 데이터가 있다면 비교 가능)
    """

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.eval()

    def extract_activation_patterns(
        self,
        dataloader,
        num_batches: int = 10
    ) -> Dict:
        """
        모델의 활동 패턴 추출

        Returns:
            patterns: 각 레이어/단계별 활동 패턴
        """
        patterns = {
            'embeddings': [],
            'step_0': [],
            'step_1': [],
            'step_2': [],
            'step_3': [],
            'final': []
        }

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting patterns", total=num_batches)):
                if batch_idx >= num_batches:
                    break

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                # 모든 step의 출력 얻기
                all_outputs = self.model(
                    input_ids,
                    attention_mask,
                    return_all_steps=True
                )

                # 각 step의 평균 활동 저장
                patterns['embeddings'].append(all_outputs[0].mean(dim=[0, 1]).cpu().numpy())

                for step_idx in range(min(4, len(all_outputs) - 1)):
                    step_key = f'step_{step_idx}'
                    patterns[step_key].append(
                        all_outputs[step_idx + 1].mean(dim=[0, 1]).cpu().numpy()
                    )

                patterns['final'].append(all_outputs[-1].mean(dim=[0, 1]).cpu().numpy())

        # 평균 패턴 계산
        for key in patterns:
            if patterns[key]:
                patterns[key] = np.mean(patterns[key], axis=0)

        return patterns

    def compare_with_brain_hypothesis(
        self,
        patterns: Dict
    ) -> Dict:
        """
        가설적 뇌 활동 패턴과 비교

        뇌과학 가설:
        - 초기 step: 넓은 활동 (탐색)
        - 중간 step: 선택적 활동 (집중)
        - 후기 step: 통합적 활동 (종합)
        """
        analysis = {}

        # 1. 활동 범위 (Activity breadth) - 표준편차로 측정
        for key, pattern in patterns.items():
            if isinstance(pattern, np.ndarray) and len(pattern) > 0:
                analysis[f'{key}_breadth'] = float(np.std(pattern))
                analysis[f'{key}_mean_activity'] = float(np.mean(np.abs(pattern)))

        # 2. 패턴 변화 (Step-to-step changes)
        step_keys = ['embeddings', 'step_0', 'step_1', 'step_2', 'step_3']
        for i in range(len(step_keys) - 1):
            if step_keys[i] in patterns and step_keys[i+1] in patterns:
                p1 = patterns[step_keys[i]]
                p2 = patterns[step_keys[i+1]]
                if isinstance(p1, np.ndarray) and isinstance(p2, np.ndarray):
                    change = np.linalg.norm(p2 - p1)
                    analysis[f'change_{step_keys[i]}_to_{step_keys[i+1]}'] = float(change)

        # 3. 가설 검증
        # 가설: 중간 step에서 가장 선택적 (낮은 breadth)
        breadths = [
            analysis.get(f'{key}_breadth', 0)
            for key in ['step_0', 'step_1', 'step_2', 'step_3']
        ]

        if breadths:
            min_breadth_idx = np.argmin(breadths)
            analysis['most_selective_step'] = int(min_breadth_idx)
            analysis['supports_selectivity_hypothesis'] = (min_breadth_idx in [1, 2])

        return analysis


def prepare_test_data(tokenizer, num_samples: int = 100):
    """테스트 데이터 준비"""
    from pnn.data.dataset import MLMDataset
    from torch.utils.data import DataLoader

    print("\n📚 Loading test data...")

    # WikiText-103 validation set
    dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-103-raw-v1",
        split="validation"
    )

    test_data = []
    for item in dataset:
        text = item['text'].strip()
        if len(text) > 20 and len(text.split()) >= 5:
            test_data.append(text)
            if len(test_data) >= num_samples:
                break

    test_dataset = MLMDataset(
        tokenizer=tokenizer,
        data=test_data,
        max_length=128,
        mask_prob=0.15
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    return test_loader


def save_visualizations(results: Dict, output_dir: Path):
    """결과 시각화 및 저장"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # MEG 결과 시각화
    if 'meg_analysis' in results:
        meg = results['meg_analysis']

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('MEG Simulation: Temporal Activity Patterns', fontsize=16)

        for idx, (key, ax) in enumerate(zip(
            ['delta_generation', 'gate_computation', 'hidden_update', 'activity_magnitude'],
            axes.flatten()
        )):
            if key in meg and 'values' in meg[key]:
                values = meg[key]['values']
                ax.plot(values, marker='o')
                ax.set_title(f'{key.replace("_", " ").title()}')
                ax.set_xlabel('Refinement Step')
                ax.set_ylabel('Activity Magnitude')
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'meg_temporal_patterns.png', dpi=300)
        plt.close()
        print(f"\n✅ Saved: {output_dir / 'meg_temporal_patterns.png'}")

    # Optogenetics 결과 시각화
    if 'optogenetics_results' in results:
        opto = results['optogenetics_results']

        # 각 컴포넌트별 억제 효과
        components = ['attention', 'gate', 'ffn']
        rates = [0, 25, 50, 75, 100]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Optogenetics Simulation: Suppression Effects', fontsize=16)

        for component in components:
            losses = []
            accs = []

            for rate in rates:
                if rate == 0:
                    key = 'baseline'
                else:
                    key = f'{component}_suppressed_{rate}'

                if key in opto:
                    losses.append(opto[key]['loss'])
                    accs.append(opto[key]['accuracy'] * 100)

            axes[0].plot(rates, losses, marker='o', label=component.capitalize())
            axes[1].plot(rates, accs, marker='o', label=component.capitalize())

        axes[0].set_xlabel('Suppression Rate (%)')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss vs Suppression Rate')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel('Suppression Rate (%)')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Accuracy vs Suppression Rate')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'optogenetics_suppression.png', dpi=300)
        plt.close()
        print(f"✅ Saved: {output_dir / 'optogenetics_suppression.png'}")

    # Brain activity 결과 시각화
    if 'brain_hypothesis_test' in results:
        brain = results['brain_hypothesis_test']

        # Activity breadth across steps
        steps = ['embeddings', 'step_0', 'step_1', 'step_2', 'step_3', 'final']
        breadths = [brain.get(f'{s}_breadth', 0) for s in steps]
        activities = [brain.get(f'{s}_mean_activity', 0) for s in steps]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Brain Activity Prediction: Pattern Analysis', fontsize=16)

        axes[0].plot(range(len(steps)), breadths, marker='o', linewidth=2)
        axes[0].set_xticks(range(len(steps)))
        axes[0].set_xticklabels(steps, rotation=45)
        axes[0].set_ylabel('Activity Breadth (std)')
        axes[0].set_title('Selectivity Across Processing Steps')
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(range(len(steps)), activities, marker='o', linewidth=2, color='orange')
        axes[1].set_xticks(range(len(steps)))
        axes[1].set_xticklabels(steps, rotation=45)
        axes[1].set_ylabel('Mean Activity')
        axes[1].set_title('Activity Level Across Processing Steps')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'brain_activity_patterns.png', dpi=300)
        plt.close()
        print(f"✅ Saved: {output_dir / 'brain_activity_patterns.png'}")


def main():
    parser = argparse.ArgumentParser(
        description='Experimental Evidence for Plastic Neural Networks'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint (e.g., checkpoints/best_model.pt)'
    )
    parser.add_argument(
        '--experiment',
        type=str,
        default='all',
        choices=['all', 'meg', 'optogenetics', 'modeling'],
        help='Which experiment to run'
    )
    parser.add_argument(
        '--num_batches',
        type=int,
        default=10,
        help='Number of batches to analyze'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/experimental_evidence',
        help='Output directory for results'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use'
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print("🧠 Experimental Evidence for Plastic Neural Networks")
    print(f"{'='*80}\n")

    # Load model
    print(f"📦 Loading checkpoint: {args.checkpoint}")

    # Create model with default config
    model_config = {
        'vocab_size': 30522,
        'hidden_size': 768,
        'num_heads': 12,
        'intermediate_size': 2048,
        'max_length': 128,
        'num_steps': 4,
        'dropout': 0.1
    }
    model = create_pnn_model(model_config)

    # Load checkpoint
    # Use weights_only=False to handle custom classes like Config
    print("📥 Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)

    # Load model state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Checkpoint might be just the state dict
        model.load_state_dict(checkpoint)

    model = model.to(args.device)
    model.eval()

    print(f"✅ Model loaded on {args.device}")

    # Load tokenizer and data
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    test_loader = prepare_test_data(tokenizer, num_samples=args.num_batches * 32)

    results = {}

    # Experiment 1: MEG Simulation
    if args.experiment in ['all', 'meg']:
        print(f"\n{'='*80}")
        print("🔬 Experiment 1: MEG Simulation (High-temporal resolution)")
        print(f"{'='*80}\n")

        meg_sim = MEGSimulator(model, args.device)
        meg_analysis = meg_sim.analyze_temporal_patterns(test_loader, args.num_batches)
        results['meg_analysis'] = meg_analysis

        print("\n📊 MEG Analysis Results:")
        for key, stats in meg_analysis.items():
            if isinstance(stats, dict):
                print(f"  {key}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")

    # Experiment 2: Optogenetics Simulation
    if args.experiment in ['all', 'optogenetics']:
        print(f"\n{'='*80}")
        print("🔬 Experiment 2: Optogenetics Simulation")
        print(f"{'='*80}\n")

        opto_sim = OptogeneticsSimulator(model, args.device)
        opto_results = opto_sim.run_suppression_experiment(test_loader, args.num_batches)
        results['optogenetics_results'] = opto_results

        print("\n📊 Optogenetics Results:")
        baseline_acc = opto_results['baseline']['accuracy']
        print(f"  Baseline: loss={opto_results['baseline']['loss']:.4f}, "
              f"acc={baseline_acc*100:.2f}%")

        for key, metrics in opto_results.items():
            if key != 'baseline':
                acc_drop = (baseline_acc - metrics['accuracy']) * 100
                print(f"  {key}: loss={metrics['loss']:.4f}, "
                      f"acc={metrics['accuracy']*100:.2f}% "
                      f"(drop: {acc_drop:.2f}%)")

    # Experiment 3: Brain Activity Modeling
    if args.experiment in ['all', 'modeling']:
        print(f"\n{'='*80}")
        print("🔬 Experiment 3: Brain Activity Modeling")
        print(f"{'='*80}\n")

        brain_predictor = BrainActivityPredictor(model, args.device)
        activation_patterns = brain_predictor.extract_activation_patterns(
            test_loader, args.num_batches
        )
        hypothesis_test = brain_predictor.compare_with_brain_hypothesis(
            activation_patterns
        )

        results['activation_patterns'] = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in activation_patterns.items()
        }
        results['brain_hypothesis_test'] = hypothesis_test

        print("\n📊 Brain Activity Analysis:")
        print(f"  Most selective step: {hypothesis_test.get('most_selective_step', 'N/A')}")
        print(f"  Supports selectivity hypothesis: "
              f"{hypothesis_test.get('supports_selectivity_hypothesis', False)}")

        for key, value in hypothesis_test.items():
            if key.endswith('_breadth'):
                print(f"  {key}: {value:.4f}")

    # Save results
    results_file = output_dir / 'experimental_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to: {results_file}")

    # Generate visualizations
    print("\n📈 Generating visualizations...")
    save_visualizations(results, output_dir)

    print(f"\n{'='*80}")
    print("✅ Experimental analysis complete!")
    print(f"{'='*80}\n")
    print(f"📁 All results saved in: {output_dir}")
    print(f"   - experimental_results.json")
    print(f"   - meg_temporal_patterns.png")
    print(f"   - optogenetics_suppression.png")
    print(f"   - brain_activity_patterns.png")


if __name__ == "__main__":
    main()
