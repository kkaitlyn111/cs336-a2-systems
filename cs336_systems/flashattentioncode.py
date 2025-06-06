import torch
import numpy as np
import einops
import triton
import triton.language as tl
import pandas as pd
import matplotlib.pyplot as plt
from flash_attention import FlashAttentionTorch as FA_Triton
from flash_attention import compiled_backward as flash_bkwd_compiled
from flash_attention import backward_pass_recomp as flash_bkwd
from flash_attention import FlashAttentionTorch as FA_Pytorch
from cs336_systems.transformer_annotated import scaled_dot_product_attention as attention


def benchmark_FA(context_length, d, dtype):
    batch_size = 1
    is_causal = True
    dtype = torch.float32
    device = torch.device('cuda')

    #forward pass
    Q = torch.randn(batch_size, context_length, d, device=device, dtype=dtype)
    K = torch.randn(batch_size, context_length, d, device=device, dtype=dtype)
    V = torch.randn(batch_size, context_length, d, device=device, dtype=dtype)

    O = torch.randn(batch_size, context_length, d, device=device, dtype=dtype)
    L = torch.randn(batch_size, context_length, device=device, dtype=dtype)
    dO = torch.randn(batch_size, context_length, d, device=device, dtype=dtype)

    def forward_triton():
        FA_Triton.apply(Q, K, V, is_causal)
    def forward_pytorch():
        mask = torch.triu(torch.ones(context_length, context_length, device=device, dtype=torch.bool), diagonal=1) if is_causal else None
        attention(Q, K, V, mask)
    def backward_triton():
        flash_bkwd_compiled(Q, K, V, O, L, dO, is_causal)
    def backward_pytorch():
        flash_bkwd(Q, K, V, O, L, dO, is_causal)
    def fb_triton():
        O = FA_Triton.apply(Q, K, V, is_causal)
        loss = torch.mean(O)
        loss.backward()
    def fb_pytorch():
        mask = torch.triu(torch.ones(context_length, context_length, device=device, dtype=torch.bool), diagonal=1) if is_causal else None
        O = attention(Q, K, V, mask)
        loss = torch.mean(O)
        loss.backward()
    
    ms_forward_triton = triton.testing.do_bench(forward_triton)
    print(f"Forward pass (Triton): {ms_forward_triton:.2f} ms")
    #do I need torch.cuda.synchronize() here?
    ms_forward_pytorch = triton.testing.do_bench(forward_pytorch)
    print(f"Forward pass (Pytorch): {ms_forward_pytorch:.2f} ms")
    ms_backward_triton = triton.testing.do_bench(backward_triton)
    print(f"Backward pass (Triton): {ms_backward_triton:.2f} ms")
    ms_backward_pytorch = triton.testing.do_bench(backward_pytorch)
    print(f"Backward pass (Pytorch): {ms_backward_pytorch:.2f} ms")

    Q.requires_grad_(True)
    K.requires_grad_(True) 
    V.requires_grad_(True)
    ms_fb_triton = triton.testing.do_bench(fb_triton)
    print(f"F-B pass (Triton): {ms_fb_triton:.2f} ms")
    ms_fb_pytorch = triton.testing.do_bench(fb_pytorch)
    print(f"F-B pass (Pytorch): {ms_fb_pytorch:.2f} ms")

    return ms_forward_triton, ms_forward_pytorch, ms_backward_triton, ms_backward_pytorch, ms_fb_triton, ms_fb_pytorch

def create_flash_attention_benchmark_table(benchmark_results, units="ms"):
    """Create and display a formatted table of Flash Attention benchmark results."""
    # Prepare data for DataFrame with multi-level index
    
    # First, collect all unique values for each dimension
    all_context_lengths = set()
    all_d_models = set()
    all_dtypes = set()
    
    for context_length in benchmark_results:
        all_context_lengths.add(context_length)
        for d in benchmark_results[context_length]:
            all_d_models.add(d)
            for dtype in benchmark_results[context_length][d]:
                all_dtypes.add(dtype)
    
    # Sort the values for consistent ordering
    all_d_models = sorted(all_d_models)
    all_dtypes = sorted(all_dtypes)
    all_context_lengths = sorted(all_context_lengths)
    
    rows = []
    # Change iteration order to: model_dim -> data_type -> context_length
    for d in all_d_models:
        for dtype in all_dtypes:
            for context_length in all_context_lengths:
                results = benchmark_results[context_length][d][dtype]
                row = {
                    'context_length': context_length,
                    'd_model': d,
                    'dtype': dtype,
                    'forward_triton': results['forward_triton'],
                    'forward_pytorch': results['forward_pytorch'],
                    'backward_triton': results['backward_triton'],
                    'backward_pytorch': results['backward_pytorch'],
                    'forward_backward_triton': results['forward_backward_triton'],
                    'forward_backward_pytorch': results['forward_backward_pytorch'],
                }
                rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    # Change index order to reflect new hierarchy: model_dim -> data_type -> context_length
    df = df.set_index(['d_model', 'dtype', 'context_length'])
    
    # Format columns with units
    metric_columns = {
        'forward_triton': f'Forward Triton ({units})',
        'forward_pytorch': f'Forward PyTorch ({units})',
        'backward_triton': f'Backward Triton ({units})',
        'backward_pytorch': f'Backward PyTorch ({units})',
        'forward_backward_triton': f'F+B Triton ({units})',
        'forward_backward_pytorch': f'F+B PyTorch ({units})'
    }
    
    # Rename columns
    df = df.rename(columns=metric_columns)
    
    # Format values with 3 decimal places
    for col in df.columns:
        df[col] = df[col].apply(lambda x: f"{x:.3f}")
    
    # Rename index names to reflect new hierarchy
    df.index.names = ['Model Dimension', 'Data Type', 'Context Length']
    
    # Set pandas display options
    pd.set_option('display.max_colwidth', 20)
    pd.set_option('display.width', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.float_format', '{:.3f}'.format)
    
    print(f"\n---------- Flash Attention Benchmark Results ----------")
    print(df.to_string(col_space=15))

    # Save as LaTeX
    latex_str = df.to_latex('flash_attention_benchmarks.tex')
    print("\nLaTeX table output:\n")
    print(latex_str)
    return df

# ---------------------------------------------------------------------------
# NEW: quick-and-dirty plotting helper
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# NEW: helper that COMBINES & SAVES required figures
# ---------------------------------------------------------------------------
def save_flash_attention_plots(benchmark_results,
                               ctx_out="latency_vs_context_length.png",
                               d_out="latency_vs_d.png",
                               bar_out="dtype_bar_comparison.png"):
    """
    Writes three images:
      • ctx_out : plots 1-2 (latency vs. context length, d=128) in one figure
      • d_out   : plots 3-4 (latency vs. d, ctx=32768) in one figure
      • bar_out : plot 5   (float32 vs bfloat16 bar chart) in one figure
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Define 3 colors for the 3 operation types
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # blue, orange, green
    
    # Group metrics by operation type
    operation_groups = [
        ("Forward", "forward_pytorch", "forward_triton"),
        ("Backward", "backward_pytorch", "backward_triton"), 
        ("F+B", "forward_backward_pytorch", "forward_backward_triton")
    ]

    # ---------- fig 1: plots 1-2 side-by-side ----------
    d_fixed = 128 #32 #128
    xs_ctx  = sorted(benchmark_results.keys())           # context lengths
    fig1, axs1 = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, dtype in zip(axs1, ("float32", "bfloat16")):
        for i, (op_name, pytorch_metric, triton_metric) in enumerate(operation_groups):
            color = colors[i]
            # Regular Attention (PyTorch) - solid line
            ys_pytorch = [benchmark_results[cl][d_fixed][dtype][pytorch_metric] for cl in xs_ctx]
            ax.plot(xs_ctx, ys_pytorch, marker="o", label=f"{op_name} Pass: Regular Attention (PyTorch)", 
                   color=color, linestyle='-')
            # Flash Attention (Triton) - dashed line
            ys_triton = [benchmark_results[cl][d_fixed][dtype][triton_metric] for cl in xs_ctx]
            ax.plot(xs_ctx, ys_triton, marker="s", label=f"{op_name} Pass: Flash Attention (Triton)", 
                   color=color, linestyle='--')
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=10)
        ax.set_xlabel("Context length")
        ax.set_title(f"{dtype}")
    axs1[0].set_ylabel("Latency (ms)")
    axs1[1].legend(loc="upper left", bbox_to_anchor=(1.04, 1.0))
    fig1.suptitle(f"Latency vs. context length (d={d_fixed})")
    fig1.tight_layout()
    fig1.savefig(ctx_out, dpi=300)
    plt.close(fig1)

    # ---------- fig 2: plots 3-4 side-by-side ----------
    cl_fixed = 32768
    xs_d     = sorted(benchmark_results[cl_fixed].keys())
    fig2, axs2 = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, dtype in zip(axs2, ("float32", "bfloat16")):
        for i, (op_name, pytorch_metric, triton_metric) in enumerate(operation_groups):
            color = colors[i]
            # Regular Attention (PyTorch) - solid line
            ys_pytorch = [benchmark_results[cl_fixed][d][dtype][pytorch_metric] for d in xs_d]
            ax.plot(xs_d, ys_pytorch, marker="o", label=f"{op_name} Pass: Regular Attention (PyTorch)", 
                   color=color, linestyle='-')
            # Flash Attention (Triton) - dashed line
            ys_triton = [benchmark_results[cl_fixed][d][dtype][triton_metric] for d in xs_d]
            ax.plot(xs_d, ys_triton, marker="s", label=f"{op_name} Pass: Flash Attention (Triton)", 
                   color=color, linestyle='--')
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=10)
        ax.set_xlabel("Model dimension d")
        ax.set_title(f"{dtype}")
    axs2[0].set_ylabel("Latency (ms)")
    axs2[1].legend(loc="upper left", bbox_to_anchor=(1.04, 1.0))
    fig2.suptitle(f"Latency vs. d (context={cl_fixed})")
    fig2.tight_layout()
    fig2.savefig(d_out, dpi=300)
    plt.close(fig2)

    # ---------- fig 3: bar plot 5 ----------
    # Update bar chart labels for clarity
    metrics = [
        "forward_triton", "forward_pytorch",
        "backward_triton", "backward_pytorch",
        "forward_backward_triton", "forward_backward_pytorch",
    ]
    labels  = [
        "Flash Attention (F)", "Regular Attention (F)",
        "Flash Attention (B)", "Regular Attention (B)",
        "Flash Attention (F+B)", "Regular Attention (F+B)",
    ]
    
    ys_f32 = [benchmark_results[cl_fixed][d_fixed]["float32"][m]  for m in metrics]
    ys_bf  = [benchmark_results[cl_fixed][d_fixed]["bfloat16"][m] for m in metrics]
    x      = np.arange(len(metrics))
    bar_w  = 0.35

    fig3, ax3 = plt.subplots(figsize=(12, 6))  # Increased width from 10 to 12
    ax3.bar(x - bar_w/2, ys_f32, width=bar_w, label="float32")
    ax3.bar(x + bar_w/2, ys_bf,  width=bar_w, label="bfloat16")
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, rotation=0, ha="center", fontsize=9)  # Added smaller font size
    ax3.set_ylabel("Latency (ms)")
    ax3.set_title(f"Flash Attention vs Regular Attention (context={cl_fixed}, d={d_fixed})")
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig(bar_out, dpi=300)
    plt.close(fig3)




#benchmarking
if __name__ == '__main__':
    context_lengths = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    ds = [16, 32, 64, 128]
    dtypes = ["float32", "bfloat16"]
    
    # Initialize nested dictionary to store results
    benchmark_results = {}
    
    for context_length in context_lengths:
        # Clear cache at the start of each new context length        
        benchmark_results[context_length] = {}
        for d in ds:
            benchmark_results[context_length][d] = {}
            for dtype in dtypes:
                print(f"\n---------- Context Length: {context_length}, d_model: {d}, dtype: {dtype} ----------")
                
                # Run benchmark and get results
                ms_forward_triton, ms_forward_pytorch, ms_backward_triton, ms_backward_pytorch, ms_fb_triton, ms_fb_pytorch = benchmark_FA(context_length, d, dtype)
                torch._dynamo.reset()

                # Store results in dictionary
                benchmark_results[context_length][d][dtype] = {
                    'forward_triton': ms_forward_triton,
                    'forward_pytorch': ms_forward_pytorch,
                    'backward_triton': ms_backward_triton,
                    'backward_pytorch': ms_backward_pytorch,
                    'forward_backward_triton': ms_fb_triton,
                    'forward_backward_pytorch': ms_fb_pytorch
                }
    
    # Create and display the results table
    create_flash_attention_benchmark_table(benchmark_results, "ms")

    save_flash_attention_plots(benchmark_results)


