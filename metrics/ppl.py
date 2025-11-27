import shutil
import subprocess


def eval_ppl(
    model_name,
    task_list=[
        "wikitext",
    ],
    output_path=".",
):
    command = "lm_eval"
    tasks = ",".join(task_list)
    args = [
        "--model",
        "hf",
        "--model_args",
        f"pretrained={model_name},cache_dir=./.cache,device_map=auto,parallelize=True",
        "--tasks",
        f"{tasks}",
        "--batch_size",
        "16",
        "--output_path",
        f"{output_path}",
    ]
    # Combine command and arguments
    full_command = [command] + args

    # Execute the command
    if shutil.which(command) is None:
        print(f"Skip ppl eval because '{command}' is not installed or not in PATH.")
        return

    # lm_eval 可能因本地 transformers 版本缺少可选模型（如 Qwen2Audio）而报错。
    # 使用 check=False 并捕获输出，避免异常向上抛出中断遗忘流程。
    proc = subprocess.run(full_command, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        print(
            "Skip ppl eval because lm_eval failed. \n"
            f"cmd: {' '.join(full_command)}\n"
            f"returncode: {proc.returncode}\n"
            f"stdout: {proc.stdout}\n"
            f"stderr: {proc.stderr}"
        )
