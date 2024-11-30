import os
import subprocess


model_paths = {
    "llava-1.5": "llava-hf/llava-1.5-7b-hf",
    "instructblip": "Salesforce/instructblip-vicuna-7b",
    "llava-next": "llava-hf/llava-v1.6-mistral-7b-hf",
}
num_gpus = 1
data_dir = "/fsx/zhuokai/maskllama/dataset/COCO"
# all_model_names = ["llava-1.5", "instructblip"]
all_model_names = ["instructblip"]
all_seeds = {
    "llava-1.5": [1, 9, 40],
    "instructblip": [26, 31, 34],
}

for model_name in all_model_names:
    for cur_seed in all_seeds[model_name]:
        for if_avg in [True, False]:
            for num_voting in [1, 2, 3, 4, 5]:
                model_path = model_paths[model_name]
                run_name = f"{model_name}_avg_{if_avg}_{num_voting}_{num_voting}_seed_{cur_seed}"
                # generate the slurm file
                script_path = f"./scripts/ablations/{run_name}.slurm"
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
                        f"#SBATCH --output=/fsx/zhuokai/maskllama/slurm/ablations/{run_name}.stdout\n",
                        f"#SBATCH --error=/fsx/zhuokai/maskllama/slurm/ablations/{run_name}.stderr\n",
                        "\n",
                        f"python -m chair_test.chair_test --seed {cur_seed} --method {run_name}_ --coco-data-dir {data_dir} --model-path {model_path} --image-numbers 500 --sample-save-name logs_ablations/{run_name}.log --model {model_name} --avg {if_avg} --voting-numbers {num_voting}\n",
                    ]
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
