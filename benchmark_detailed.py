#!/usr/bin/env python3
"""
Detailed Benchmark Script with Operator-level Profiling
Measures execution time and percentages for each operator/layer in the model.
"""

import argparse
import time
import sys
import os
import numpy as np


class TorchTimer:
    def __init__(self):
        import torch
        self.torch = torch
        if torch.cuda.is_available():
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
        else:
            self.start_event = None
            self._start_time = None

    def start(self):
        if self.start_event is not None:
            self.start_event.record()
        else:
            self._start_time = time.perf_counter()

    def stop(self):
        if self.start_event is not None:
            self.end_event.record()
            self.end_event.synchronize()
            return self.start_event.elapsed_time(self.end_event)
        return (time.perf_counter() - self._start_time) * 1000


def detailed_profile_torch(model, input_features, input_ids, input_features_mask, num_runs=3):
    import torch
    results = {}
    timer = TorchTimer()

    print("\n" + "=" * 70)
    print("1. DETAILED MACRO COMPONENT PROFILING")
    print("=" * 70)

    # [1/4] Audio Encoder
    print("\n[1/4] Profiling Audio Encoder...")
    encoder_times = []
    # Warmup
    for _ in range(2): model.audio_encoder(input_features)
    for _ in range(num_runs):
        if torch.cuda.is_available(): torch.cuda.synchronize()
        timer.start()
        audio_features = model.audio_encoder(input_features)
        elapsed = timer.stop()
        encoder_times.append(elapsed)
    results['audio_encoder'] = {'mean': np.mean(encoder_times), 'std': np.std(encoder_times)}
    print(f"  Audio Encoder: {results['audio_encoder']['mean']:.2f}ms (+/- {results['audio_encoder']['std']:.2f}ms)")

    # [2/4] Multi-modal Projector
    print("\n[2/4] Profiling Multi-modal Projector...")
    projector_times = []
    for _ in range(2): model.multi_modal_projector(audio_features)
    for _ in range(num_runs):
        if torch.cuda.is_available(): torch.cuda.synchronize()
        timer.start()
        projected = model.multi_modal_projector(audio_features)
        elapsed = timer.stop()
        projector_times.append(elapsed)
    results['projector'] = {'mean': np.mean(projector_times), 'std': np.std(projector_times)}
    print(f"  Projector: {results['projector']['mean']:.2f}ms (+/- {results['projector']['std']:.2f}ms)")

    # [3/4] Text Decoder (Prefill)
    print("\n[3/4] Profiling Text Decoder (Prefill)...")
    embed_tokens = model.text_decoder.embed_tokens
    text_embeds = embed_tokens(input_ids)
    combined_embeds = text_embeds.clone()

    # Fake audio token injection for shape matching
    audio_mask = (input_ids == 59260)
    if torch.any(audio_mask):
        audio_positions = torch.where(audio_mask[0])[0]
        num_audio_tokens = int(audio_positions.numel())
        if num_audio_tokens <= projected.shape[1]:
            combined_embeds[0, audio_positions[:projected.shape[1]]] = projected[0, :num_audio_tokens]

    prefill_times = []
    for _ in range(2): model.text_decoder(inputs_embeds=combined_embeds)
    for _ in range(num_runs):
        if torch.cuda.is_available(): torch.cuda.synchronize()
        timer.start()
        hidden_states = model.text_decoder(inputs_embeds=combined_embeds)
        elapsed = timer.stop()
        prefill_times.append(elapsed)
    results['decoder_prefill'] = {'mean': np.mean(prefill_times), 'std': np.std(prefill_times)}
    print(
        f"  Decoder Prefill: {results['decoder_prefill']['mean']:.2f}ms (+/- {results['decoder_prefill']['std']:.2f}ms)")

    # [4/4] Decode Steps (Autoregressive)
    print("\n[4/4] Profiling Decode Steps (per token)...")
    decode_times = []
    num_decode_steps = 10
    logits = model.lm_head(hidden_states[:, -1:, :])
    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    next_embed = embed_tokens(next_token)

    for _ in range(2): model.text_decoder(inputs_embeds=next_embed)
    for _ in range(num_decode_steps):
        if torch.cuda.is_available(): torch.cuda.synchronize()
        timer.start()
        step_hidden = model.text_decoder(inputs_embeds=next_embed)
        step_logits = model.lm_head(step_hidden)
        # mock next token
        elapsed = timer.stop()
        decode_times.append(elapsed)

    results['decode_step'] = {'mean': np.mean(decode_times), 'std': np.std(decode_times)}
    print(f"  Single Decode Step: {results['decode_step']['mean']:.2f}ms (+/- {results['decode_step']['std']:.2f}ms)")

    return results


def profile_attention_ops_torch(seq_len=256, num_runs=5):
    import torch
    print("\n" + "=" * 70)
    print("2. MICRO BENCHMARK: ATTENTION (TORCH vs TRITON)")
    print("=" * 70)

    timer = TorchTimer()
    results = {}
    hidden_size, num_heads = 2048, 16
    head_dim = hidden_size // num_heads
    batch_size = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

    print(f"\nSequence length: {seq_len}")

    # Torch Standard
    print("\n[1] Standard Attention (Torch Matmul)...")
    q_2d = q.reshape(batch_size * num_heads, seq_len, head_dim)
    k_2d = k.reshape(batch_size * num_heads, seq_len, head_dim)
    v_2d = v.reshape(batch_size * num_heads, seq_len, head_dim)

    # Warmup
    for _ in range(3):
        scores = torch.matmul(q_2d, k_2d.transpose(1, 2)) / torch.sqrt(
            torch.tensor(head_dim, dtype=torch.float32, device=device))
        attn_weights = torch.exp(scores - torch.max(scores, dim=-1, keepdim=True).values)
        output = torch.matmul(attn_weights / torch.sum(attn_weights, dim=-1, keepdim=True), v_2d)

    matmul_times = []
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats(device)
    for _ in range(num_runs):
        if torch.cuda.is_available(): torch.cuda.synchronize()
        timer.start()
        scores = torch.matmul(q_2d, k_2d.transpose(1, 2)) / torch.sqrt(
            torch.tensor(head_dim, dtype=torch.float32, device=device))
        attn_weights = torch.exp(scores - torch.max(scores, dim=-1, keepdim=True).values)
        attn_weights = attn_weights / torch.sum(attn_weights, dim=-1, keepdim=True)
        output = torch.matmul(attn_weights, v_2d)
        matmul_times.append(timer.stop())

    mem_torch = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if torch.cuda.is_available() else 0
    results['matmul_attention'] = np.mean(matmul_times)
    print(
        f"  Torch matmul: {np.mean(matmul_times):.2f}ms (+/- {np.std(matmul_times):.2f}ms) | Peak VRAM: {mem_torch:.1f} MB")

    # Triton Fused
    print("\n[2] Custom Triton Fused RoPE+FlashAttention...")
    try:
        from rope import RotaryEmbedding
        from attention import scaled_dot_product_attention_with_rope
        rope = RotaryEmbedding(dim=head_dim, max_position_embeddings=seq_len + 100)
        cos, sin = rope(q)

        for _ in range(3): _ = scaled_dot_product_attention_with_rope(q, k, v, cos, sin)  # Warmup

        triton_times = []
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats(device)
        for _ in range(num_runs):
            if torch.cuda.is_available(): torch.cuda.synchronize()
            timer.start()
            _ = scaled_dot_product_attention_with_rope(q, k, v, cos, sin)
            triton_times.append(timer.stop())

        mem_triton = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if torch.cuda.is_available() else 0
        results['triton_fused_attention'] = np.mean(triton_times)
        print(
            f"  Triton Fused: {np.mean(triton_times):.2f}ms (+/- {np.std(triton_times):.2f}ms) | Peak VRAM: {mem_triton:.1f} MB")
        print(f"  -> Speedup vs Torch: {np.mean(matmul_times) / np.mean(triton_times):.2f}x")
    except Exception as e:
        print(f"  [Skipped]: {e}")
    return results


def profile_linear_ops_torch(hidden_size=2048, intermediate_size=5632, batch_size=1, seq_len=256, num_runs=5):
    import torch
    print("\n" + "=" * 70)
    print("3. MICRO BENCHMARK: MLP/GEMM (TORCH vs TRITON)")
    print("=" * 70)

    timer = TorchTimer()
    results = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(batch_size, seq_len, hidden_size, device=device)

    # Torch Full MLP
    w_gate = torch.randn(hidden_size, intermediate_size, device=device)
    w_up = torch.randn(hidden_size, intermediate_size, device=device)
    w_down = torch.randn(intermediate_size, hidden_size, device=device)

    print("\n[1] Torch Full MLP (SwiGLU style)...")
    for _ in range(3):
        gate = torch.matmul(x, w_gate)
        output = torch.matmul((gate * (1 / (1 + torch.exp(-gate)))) * torch.matmul(x, w_up), w_down)

    mlp_times = []
    for _ in range(num_runs):
        if torch.cuda.is_available(): torch.cuda.synchronize()
        timer.start()
        gate = torch.matmul(x, w_gate)
        up = torch.matmul(x, w_up)
        hidden = (gate * (1 / (1 + torch.exp(-gate)))) * up
        output = torch.matmul(hidden, w_down)
        mlp_times.append(timer.stop())

    results['torch_mlp'] = np.mean(mlp_times)
    print(f"  Torch MLP: {np.mean(mlp_times):.2f}ms (+/- {np.std(mlp_times):.2f}ms)")

    # Triton Fused MLP
    print("\n[2] Custom Triton Fused SwiGLU MLP...")
    try:
        from layers import MLP
        triton_mlp = MLP(hidden_size, intermediate_size, activation="silu", use_gating=True)
        triton_mlp.gate_proj.weight.data = w_gate.t().contiguous()
        triton_mlp.up_proj.weight.data = w_up.t().contiguous()
        triton_mlp.down_proj.weight.data = w_down.t().contiguous()

        for _ in range(3): _ = triton_mlp(x)  # Warmup

        triton_mlp_times = []
        for _ in range(num_runs):
            if torch.cuda.is_available(): torch.cuda.synchronize()
            timer.start()
            _ = triton_mlp(x)
            triton_mlp_times.append(timer.stop())

        results['triton_fused_mlp'] = np.mean(triton_mlp_times)
        print(f"  Triton Fused MLP: {np.mean(triton_mlp_times):.2f}ms (+/- {np.std(triton_mlp_times):.2f}ms)")
        print(f"  -> Speedup vs Torch: {np.mean(mlp_times) / np.mean(triton_mlp_times):.2f}x")
    except Exception as e:
        print(f"  [Skipped]: {e}")
    return results


def print_summary(component_results):
    print("\n" + "=" * 70)
    print("4. PERFORMANCE SUMMARY (EXECUTION PERCENTAGE)")
    print("=" * 70)
    print("\n{:<30} {:>15} {:>15}".format("Component", "Time (ms)", "% of Total"))
    print("-" * 62)

    total = 0
    if component_results:
        # 假设一次典型的推理包含 1次 Encoder, 1次 Projector, 1次 Prefill, 和 50次 Decode step
        total += component_results.get('audio_encoder', {}).get('mean', 0)
        total += component_results.get('projector', {}).get('mean', 0)
        total += component_results.get('decoder_prefill', {}).get('mean', 0)
        decode_total = component_results.get('decode_step', {}).get('mean', 0) * 50
        total += decode_total

        for key, label in [
            ('audio_encoder', 'Audio Encoder'),
            ('projector', 'Projector'),
            ('decoder_prefill', 'Decoder (Prefill)'),
        ]:
            if key in component_results:
                t = component_results[key]['mean']
                pct = (t / total * 100) if total > 0 else 0
                print(f"{label:<30} {t:>13.2f}ms {pct:>14.1f}%")

        if 'decode_step' in component_results:
            pct = (decode_total / total * 100) if total > 0 else 0
            print(f"{'Decoder (50 steps)':<30} {decode_total:>13.2f}ms {pct:>14.1f}%")

    print("-" * 62)
    print(f"{'TOTAL (Estimated for 50 tokens)':<30} {total:>13.2f}ms")
    print("\n* Note: Total time is simulated based on 50 auto-regressive decode steps.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=str, help='Folder name to benchmark')
    parser.add_argument('--seq-len', type=int, default=1024, help='Sequence length for micro-benchmarks')
    parser.add_argument('--runs', type=int, default=5, help='Number of profiling runs')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(script_dir, args.folder)
    sys.path.insert(0, folder_path)

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading model and creating dummy inputs for Macro profiling...")
    from weight_loader import load_model_from_hf
    model, _ = load_model_from_hf("zai-org/GLM-ASR-Nano-2512")

    # Create dummy inputs to bypass external audio dependencies and focus purely on profiling
    dummy_audio = torch.randn(1, 128, 500, device=device)
    dummy_ids = torch.tensor([[59253, 10, 59261] + [59260] * 125 + [59262, 59253, 10]], dtype=torch.int64,
                             device=device)

    # 1. Macro Breakdown
    comp_results = detailed_profile_torch(model, dummy_audio, dummy_ids, None, num_runs=args.runs)

    # 2 & 3. Micro Benchmarks
    _ = profile_attention_ops_torch(seq_len=args.seq_len, num_runs=args.runs)
    _ = profile_linear_ops_torch(seq_len=args.seq_len, num_runs=args.runs)

    # 4. Final Summary
    print_summary(comp_results)

    sys.path.remove(folder_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())