"""
Experimental Evidence for Hierarchical PNN
============================================

Hierarchical PNN (DeltaRefinerHierarchical) 전용 실험 스크립트

실험 종류:
1. MEG Simulation - Gamma cycle 내 활동 패턴
2. Block Analysis - 각 hierarchical block의 기여도
3. Gate Analysis - Mini-gate들의 선택성 분석
4. Mountain Effect - Mountain-shaped FFN의 효과 검증

Usage:
    python scripts/experimental_evidence_hierarchical.py \
        --checkpoint /content/drive/MyDrive/pnn/hierarchical/best_model.pt \
        --intermediate_size "640,896,1024,896,640" \
        --num_blocks 5
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

from pnn.models.pnn import create_pnn_model
from transformers import BertTokenizer
from datasets import load_dataset


class HierarchicalMEGSimulator:
    """
    Hierarchical PNN용 MEG 시뮬레이션
    각 block의 temporal dynamics 분석
    """

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.eval()

        # Hierarchical refiner인지 확인
        if not hasattr(model.delta_refiner, 'blocks'):
            raise ValueError("This simulator requires a hierarchical PNN model")

        self.num_blocks = len(model.delta_refiner.blocks)
        self.activations = {}
        self.register_hooks()

    def register_hooks(self):
        """각 block의 활동 기록"""
        def get_activation(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    output = output[0]
                self.activations[name] = output.detach()
            return hook

        # 각 block의 attention, ffn에 hook 등록
        for i, block in enumerate(self.model.delta_refiner.blocks):
            block['attention'].register_forward_hook(
                get_activation(f'block_{i}_attention')
            )
            block['ffn'].register_forward_hook(
                get_activation(f'block_{i}_ffn')
            )

        # Mini-gates에 hook 등록
        for i, gate in enumerate(self.model.delta_refiner.mini_gates):
            gate.register_forward_hook(
                get_activation(f'mini_gate_{i}')
            )

    def analyze_temporal_patterns(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_batches: int = 10
    ) -> Dict:
        """
        Temporal pattern 분석
        각 refinement step에서 block들의 활동 패턴 측정
        """
        results = {
            'step_block_activities': [],  # [num_steps, num_blocks]
            'step_gate_values': [],       # [num_steps, num_blocks]
            'block_contributions': [],     # 각 block의 기여도
        }

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="MEG Analysis")):
                if batch_idx >= num_batches:
                    break

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

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

                # 각 step별 분석
                step_activities = []
                step_gates = []

                for step in range(self.model.num_steps):
                    self.activations.clear()

                    # Delta refiner forward
                    delta = self.model.delta_refiner(hidden, attn_mask)

                    # Block별 활동 수집
                    block_activities = []
                    gate_values = []

                    for i in range(self.num_blocks):
                        # FFN 활동
                        ffn_key = f'block_{i}_ffn'
                        if ffn_key in self.activations:
                            activity = self.activations[ffn_key].abs().mean().cpu().item()
                            block_activities.append(activity)

                        # Gate 값
                        gate_key = f'mini_gate_{i}'
                        if gate_key in self.activations:
                            gate_val = self.activations[gate_key].mean().cpu().item()
                            gate_values.append(gate_val)

                    step_activities.append(block_activities)
                    step_gates.append(gate_values)

                    hidden = hidden + delta

                results['step_block_activities'].append(step_activities)
                results['step_gate_values'].append(step_gates)

        # 평균 계산
        step_block_act = np.array(results['step_block_activities'])  # [batches, steps, blocks]
        step_gate_val = np.array(results['step_gate_values'])

        results['mean_block_activities'] = step_block_act.mean(axis=0)  # [steps, blocks]
        results['mean_gate_values'] = step_gate_val.mean(axis=0)

        return results


class BlockContributionAnalyzer:
    """
    각 hierarchical block의 기여도 분석
    - Block ablation (특정 block 제거)
    - Mountain effect 검증 (중간 block이 중요한지)
    """

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.eval()

        if not hasattr(model.delta_refiner, 'blocks'):
            raise ValueError("This analyzer requires a hierarchical PNN model")

        self.num_blocks = len(model.delta_refiner.blocks)

    def analyze_block_importance(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_batches: int = 10
    ) -> Dict:
        """
        각 block을 제거했을 때 성능 변화 측정
        mini_gate를 0으로 만들어서 block 기여도 제거
        """
        results = {
            'baseline': {'loss': 0.0, 'accuracy': 0.0},
            'ablations': {}
        }

        # Baseline 성능
        print("  Measuring baseline performance...")
        baseline_metrics = self._evaluate(dataloader, num_batches)
        results['baseline'] = baseline_metrics

        # 각 block ablation
        for block_idx in range(self.num_blocks):
            print(f"  Ablating block {block_idx}...")

            # mini_gate를 비활성화 (항상 0 출력)
            # Hook을 사용해서 gate 출력을 0으로 변경
            def zero_gate_hook(module, input, output):
                return torch.zeros_like(output)

            handle = self.model.delta_refiner.mini_gates[block_idx].register_forward_hook(zero_gate_hook)

            # 평가
            ablation_metrics = self._evaluate(dataloader, num_batches)
            results['ablations'][f'block_{block_idx}'] = ablation_metrics

            # Hook 제거 (복구)
            handle.remove()

        # 기여도 계산 (baseline - ablation)
        results['contributions'] = {}
        for block_name, metrics in results['ablations'].items():
            loss_increase = metrics['loss'] - baseline_metrics['loss']
            acc_decrease = baseline_metrics['accuracy'] - metrics['accuracy']
            results['contributions'][block_name] = {
                'loss_increase': loss_increase,
                'accuracy_decrease': acc_decrease
            }

        return results

    def _evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_batches: int
    ) -> Dict:
        """모델 평가"""
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= num_batches:
                    break

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                # Forward pass
                hidden = self.model(input_ids, attention_mask)
                loss, logits = self.model.get_mlm_loss(hidden, labels)

                # Accuracy
                predictions = logits.argmax(dim=-1)
                mask = labels != -100
                correct = (predictions == labels) & mask

                total_loss += loss.item()
                total_correct += correct.sum().item()
                total_tokens += mask.sum().item()

        avg_loss = total_loss / num_batches
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0

        return {'loss': avg_loss, 'accuracy': accuracy}


class GateSpecificityAnalyzer:
    """
    Mini-gate들의 선택성 분석
    - 각 gate가 얼마나 선택적인지 (entropy 측정)
    - Block별 gate pattern 차이
    """

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.eval()

        if not hasattr(model.delta_refiner, 'mini_gates'):
            raise ValueError("This analyzer requires a hierarchical PNN model")

        self.num_blocks = len(model.delta_refiner.mini_gates)
        self.gate_values = {i: [] for i in range(self.num_blocks)}

    def analyze_gate_patterns(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_batches: int = 10
    ) -> Dict:
        """Gate pattern 분석"""

        # Hook 등록
        def get_gate_hook(gate_idx):
            def hook(module, input, output):
                self.gate_values[gate_idx].append(output.detach().cpu())
            return hook

        handles = []
        for i, gate in enumerate(self.model.delta_refiner.mini_gates):
            handle = gate.register_forward_hook(get_gate_hook(i))
            handles.append(handle)

        # Data 수집
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Gate Analysis")):
                if batch_idx >= num_batches:
                    break

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                _ = self.model(input_ids, attention_mask)

        # Hook 제거
        for handle in handles:
            handle.remove()

        # 분석
        results = {
            'gate_statistics': {},
            'gate_entropy': {},
            'gate_sparsity': {}
        }

        for gate_idx in range(self.num_blocks):
            gate_vals = torch.cat(self.gate_values[gate_idx], dim=0)  # [total_tokens, hidden_size]

            # Statistics
            results['gate_statistics'][f'gate_{gate_idx}'] = {
                'mean': gate_vals.mean().item(),
                'std': gate_vals.std().item(),
                'min': gate_vals.min().item(),
                'max': gate_vals.max().item()
            }

            # Entropy (per dimension)
            # Higher entropy = less selective
            probs = gate_vals.mean(dim=0)  # Average across tokens
            probs = torch.clamp(probs, 1e-10, 1.0)
            entropy = -(probs * torch.log(probs) + (1-probs) * torch.log(1-probs)).mean()
            results['gate_entropy'][f'gate_{gate_idx}'] = entropy.item()

            # Sparsity (얼마나 많은 dimension이 거의 0인지)
            threshold = 0.1
            sparsity = (gate_vals.mean(dim=0) < threshold).float().mean()
            results['gate_sparsity'][f'gate_{gate_idx}'] = sparsity.item()

        return results


def prepare_test_data(tokenizer, num_samples=320, max_length=128):
    """테스트 데이터 준비"""
    from pnn.data.dataset import MLMDataset

    print("📚 Loading test data...")
    dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-103-raw-v1",
        split="validation",
        streaming=True
    )

    samples = []
    for item in dataset:
        text = item['text'].strip()
        if len(text) > 20:
            samples.append(text)
            if len(samples) >= num_samples:
                break

    mlm_dataset = MLMDataset(
        tokenizer,
        samples,
        max_length=max_length,
        mask_prob=0.15
    )

    dataloader = torch.utils.data.DataLoader(
        mlm_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2
    )

    return dataloader


def visualize_results(results: Dict, output_dir: Path):
    """결과 시각화"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Block activities over steps
    if 'meg_analysis' in results:
        meg_data = results['meg_analysis']
        if 'mean_block_activities' in meg_data:
            activities = meg_data['mean_block_activities']  # [steps, blocks]

            plt.figure(figsize=(10, 6))
            for block_idx in range(activities.shape[1]):
                plt.plot(activities[:, block_idx], marker='o', label=f'Block {block_idx}')
            plt.xlabel('Refinement Step')
            plt.ylabel('Activity Magnitude')
            plt.title('Block Activities Across Refinement Steps')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(output_dir / 'block_activities.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  💾 Saved: {output_dir / 'block_activities.png'}")

    # 2. Block contributions (ablation study)
    if 'block_analysis' in results:
        block_data = results['block_analysis']
        if 'contributions' in block_data:
            contributions = block_data['contributions']

            block_names = list(contributions.keys())
            acc_decreases = [contributions[b]['accuracy_decrease'] for b in block_names]

            plt.figure(figsize=(10, 6))
            bars = plt.bar(range(len(block_names)), acc_decreases)

            # Color bars - mountain pattern
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(block_names)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)

            plt.xlabel('Block')
            plt.ylabel('Accuracy Decrease (when ablated)')
            plt.title('Block Importance (Higher = More Important)')
            plt.xticks(range(len(block_names)), block_names, rotation=45)
            plt.grid(True, alpha=0.3, axis='y')
            plt.savefig(output_dir / 'block_contributions.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  💾 Saved: {output_dir / 'block_contributions.png'}")

    # 3. Gate statistics
    if 'gate_analysis' in results:
        gate_data = results['gate_analysis']
        if 'gate_statistics' in gate_data:
            stats = gate_data['gate_statistics']

            gate_names = list(stats.keys())
            means = [stats[g]['mean'] for g in gate_names]
            stds = [stats[g]['std'] for g in gate_names]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            # Mean values
            ax1.bar(range(len(gate_names)), means, color='skyblue')
            ax1.set_xlabel('Gate')
            ax1.set_ylabel('Mean Gate Value')
            ax1.set_title('Average Gate Activation')
            ax1.set_xticks(range(len(gate_names)))
            ax1.set_xticklabels(gate_names, rotation=45)
            ax1.grid(True, alpha=0.3, axis='y')

            # Entropy
            if 'gate_entropy' in gate_data:
                entropies = [gate_data['gate_entropy'][g] for g in gate_names]
                ax2.bar(range(len(gate_names)), entropies, color='coral')
                ax2.set_xlabel('Gate')
                ax2.set_ylabel('Entropy')
                ax2.set_title('Gate Selectivity (Lower = More Selective)')
                ax2.set_xticks(range(len(gate_names)))
                ax2.set_xticklabels(gate_names, rotation=45)
                ax2.grid(True, alpha=0.3, axis='y')

            plt.tight_layout()
            plt.savefig(output_dir / 'gate_analysis.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  💾 Saved: {output_dir / 'gate_analysis.png'}")


def main():
    parser = argparse.ArgumentParser(
        description='Experimental Evidence for Hierarchical PNN'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to hierarchical PNN checkpoint'
    )
    parser.add_argument(
        '--intermediate_size',
        type=str,
        required=True,
        help='Comma-separated list of intermediate sizes (e.g., "640,896,1024,896,640")'
    )
    parser.add_argument(
        '--num_blocks',
        type=int,
        required=True,
        help='Number of hierarchical blocks'
    )
    parser.add_argument(
        '--experiment',
        type=str,
        default='all',
        choices=['all', 'meg', 'blocks', 'gates'],
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
        default='results/hierarchical_evidence',
        help='Output directory'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device'
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print("🧠 Experimental Evidence for Hierarchical PNN")
    print(f"{'='*80}\n")

    # Parse intermediate_size
    intermediate_sizes = [int(x.strip()) for x in args.intermediate_size.split(',')]
    assert len(intermediate_sizes) == args.num_blocks, \
        f"intermediate_size list length ({len(intermediate_sizes)}) must match num_blocks ({args.num_blocks})"

    print(f"📦 Loading checkpoint: {args.checkpoint}")
    print(f"   Intermediate sizes: {intermediate_sizes}")
    print(f"   Num blocks: {args.num_blocks}")

    # Create hierarchical model
    model_config = {
        'vocab_size': 30522,
        'hidden_size': 768,
        'num_heads': 12,
        'intermediate_size': intermediate_sizes,
        'max_length': 128,
        'num_steps': 4,
        'num_blocks': args.num_blocks,
        'use_checkpoint': False,
        'dropout': 0.1
    }

    model = create_pnn_model(model_config, model_type='pnn_hierarchical')

    # Load checkpoint
    print("📥 Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Remove _orig_mod. prefix if present
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('_orig_mod.', '')
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict, strict=True)
    model = model.to(args.device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model loaded: {total_params:,} parameters ({total_params/1e6:.1f}M)")

    if 'epoch' in checkpoint:
        print(f"   Epoch: {checkpoint['epoch']}")
    if 'best_acc' in checkpoint:
        print(f"   Best Acc: {checkpoint['best_acc']:.4f} ({checkpoint['best_acc']*100:.2f}%)")

    # Load data
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    test_loader = prepare_test_data(tokenizer, num_samples=args.num_batches * 32)

    results = {}

    # Experiment 1: MEG Analysis
    if args.experiment in ['all', 'meg']:
        print(f"\n{'='*80}")
        print("🔬 Experiment 1: MEG Simulation (Temporal Dynamics)")
        print(f"{'='*80}\n")

        meg_sim = HierarchicalMEGSimulator(model, args.device)
        meg_results = meg_sim.analyze_temporal_patterns(test_loader, args.num_batches)
        results['meg_analysis'] = meg_results

        print("\n📊 MEG Results:")
        activities = meg_results['mean_block_activities']
        print(f"   Block activities shape: {activities.shape} (steps × blocks)")
        print(f"   Activities by step:")
        for step_idx in range(activities.shape[0]):
            step_acts = activities[step_idx]
            print(f"     Step {step_idx}: {[f'{x:.4f}' for x in step_acts]}")

    # Experiment 2: Block Importance
    if args.experiment in ['all', 'blocks']:
        print(f"\n{'='*80}")
        print("🔬 Experiment 2: Block Contribution Analysis")
        print(f"{'='*80}\n")

        block_analyzer = BlockContributionAnalyzer(model, args.device)
        block_results = block_analyzer.analyze_block_importance(test_loader, args.num_batches)
        results['block_analysis'] = block_results

        print("\n📊 Block Ablation Results:")
        baseline = block_results['baseline']
        print(f"   Baseline: loss={baseline['loss']:.4f}, acc={baseline['accuracy']*100:.2f}%")
        print(f"\n   Ablation impacts:")
        for block_name, contrib in block_results['contributions'].items():
            print(f"     {block_name}: Δacc={contrib['accuracy_decrease']*100:+.2f}%, "
                  f"Δloss={contrib['loss_increase']:+.4f}")

    # Experiment 3: Gate Analysis
    if args.experiment in ['all', 'gates']:
        print(f"\n{'='*80}")
        print("🔬 Experiment 3: Gate Specificity Analysis")
        print(f"{'='*80}\n")

        gate_analyzer = GateSpecificityAnalyzer(model, args.device)
        gate_results = gate_analyzer.analyze_gate_patterns(test_loader, args.num_batches)
        results['gate_analysis'] = gate_results

        print("\n📊 Gate Analysis Results:")
        for gate_name, stats in gate_results['gate_statistics'].items():
            entropy = gate_results['gate_entropy'].get(gate_name, 0)
            sparsity = gate_results['gate_sparsity'].get(gate_name, 0)
            print(f"   {gate_name}:")
            print(f"     Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
            print(f"     Entropy: {entropy:.4f} (lower = more selective)")
            print(f"     Sparsity: {sparsity*100:.1f}% (dims < 0.1)")

    # Save results
    results_file = output_dir / 'results.json'

    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj

    serializable_results = convert_to_serializable(results)

    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"\n💾 Results saved to: {results_file}")

    # Visualize
    print("\n📊 Generating visualizations...")
    visualize_results(results, output_dir)

    print(f"\n{'='*80}")
    print("✅ Analysis Complete!")
    print(f"   Results: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
