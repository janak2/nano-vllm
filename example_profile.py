import os
import torch
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
from pathlib import Path

TRACES_DIR = Path(__file__).parent / "traces"


def main(): 
    path = snapshot_download("Qwen/Qwen3-0.6B", local_dir=Path(__file__).parent.parent)
    print(path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    sampling_params = [SamplingParams(temperature=0.6, max_tokens=10), SamplingParams(temperature=0.6, max_tokens=20)]
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]

    TRACES_DIR.mkdir(exist_ok=True)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        with_stack=True,
    ) as prof:
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

    prof.export_chrome_trace(str(TRACES_DIR / "trace_bs_2_no_tqdm.json.gz"))
    print(f"Trace saved to {TRACES_DIR / 'trace_bs_2_no_tqdm.json.gz'}")

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
