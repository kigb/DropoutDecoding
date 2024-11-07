import os
import subprocess


model_paths = {
    "llava-1.5": "llava-hf/llava-1.5-7b-hf",
    "instructblip": "Salesforce/instructblip-vicuna-7b",
    "llava-next": "llava-hf/llava-v1.6-mistral-7b-hf",
}
num_gpus = 1
all_methods = ["original", "uncertainty-guided"]
data_dir = "/fsx/zhuokai/maskllama/dataset/COCO"
all_seeds = [6, 7, 8, 9, 10]
all_model_names = ["llava-1.5", "instructblip", "llava-next"]
# all_model_names = ["instructblip"]

for cur_method in all_methods:
    for cur_seed in all_seeds:
        for model_name in all_model_names:
            model_path = model_paths[model_name]
            run_name = f"{cur_method}_{model_name}_seed_{cur_seed}"
            # generate the slurm file
            script_path = f"./scripts/main_experiments/{run_name}.slurm"
            script_dir = os.path.dirname(script_path)
            if not os.path.exists(script_dir):
                os.makedirs(script_dir, exist_ok=True)

            with open(script_path, "w") as f:
                lines_to_write = [
                    "#!/bin/bash\n",
                    "#\n",
                    "#SBATCH --mail-user=zhuokai@meta.com\n",
                    "#SBATCH --mail-type=ALL\n",
                    "#SBATCH --chdir=/fsx/zhuokai/maskllama/\n",
                    f"#SBATCH --gres=gpu:{num_gpus}\n",
                    "#SBATCH --mem 512G\n",
                    "#SBATCH --time 0-12:00:00\n",
                    "#SBATCH -c 64\n",
                    f"#SBATCH --job-name={run_name}\n",
                    f"#SBATCH --output=/fsx/zhuokai/maskllama/slurm/main_experiments/{run_name}.stdout\n",
                    f"#SBATCH --error=/fsx/zhuokai/maskllama/slurm/main_experiments/{run_name}.stderr\n",
                    "\n",
                    f"python -m chair_test.chair_test --seed {cur_seed} --method {run_name}_ --coco-data-dir {data_dir} --model-path {model_path} --image-numbers 500 --sample-save-name logs/{run_name}.log --model {model_name}",
                ]
                if cur_method == "original":
                    lines_to_write.append(" --original True\n")
                else:
                    lines_to_write.append("\n")
                for cur_line in lines_to_write:
                    f.write(cur_line)
                f.close()

            subprocess.run(
                [
                    "sbatch",
                    f"{script_path}",
                ]
            )
            print(f"Submitted task for {run_name}\n")
