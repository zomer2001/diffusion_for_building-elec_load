import os
import pandas as pd
import yaml

# 定义路径和参数
train_data_dir = 'Data/train_data'
config_file_path = 'ori_train_diffts.yaml'
sample_config_path = 'sample.yaml'
results_base_dir = 'CK/diffts'
output_base_dir = 'OUTPUT/diffts_gen'
sparsity_rates = [0.9,0.8,0.7,0.5,0.4,0.3,0.2]
subfolders = ['720','4320']

os.makedirs(train_data_dir, exist_ok=True)
os.makedirs(results_base_dir, exist_ok=True)
os.makedirs(output_base_dir, exist_ok=True)

def generate_csv_files():
    with open(config_file_path, 'r') as file:
        config = yaml.safe_load(file)

    for subfolder in subfolders:
        input_dir = os.path.join(train_data_dir, subfolder)
        if not os.path.exists(input_dir):
            print(f"子文件夹 {input_dir} 不存在，跳过。")
            continue

        for filename in os.listdir(input_dir):
            if filename.endswith('.csv'):
                filepath = os.path.join(input_dir, filename)
                dataset_name = filename.replace('.csv', '')
                df = pd.read_csv(filepath)

                for sparsity in sparsity_rates:
                    sparsity_percentage = int(sparsity * 100)
                    sparsity_folder = os.path.join(train_data_dir, f'{sparsity_percentage}')
                    os.makedirs(sparsity_folder, exist_ok=True)

                    results_sparsity_folder = os.path.join(results_base_dir, f'{sparsity_percentage}', 'train')
                    os.makedirs(results_sparsity_folder, exist_ok=True)

                    sparsity_train_data_folder = os.path.join(sparsity_folder, 'train')
                    os.makedirs(sparsity_train_data_folder, exist_ok=True)

                    train_data_file_path = os.path.join(sparsity_train_data_folder, filename)
                    df.to_csv(train_data_file_path, index=False)
                    print(f"训练数据已生成：{train_data_file_path}（稀疏率：{sparsity_percentage}）")

                    data_root = os.path.abspath(train_data_file_path)
                    results_folder = os.path.join(results_sparsity_folder, dataset_name)
                    results_foldertag = os.path.join(results_sparsity_folder, f'{dataset_name}_24')

                    output_path = os.path.join(output_base_dir, str(sparsity_percentage), dataset_name)

                    if not os.path.exists(results_foldertag):
                        print(f"文件夹 {results_foldertag} 不存在，开始训练。")
                        start_training(config, data_root, results_folder, sparsity, dataset_name, sparsity_percentage)
                        run_sampling(data_root,dataset_name, sparsity_percentage, results_folder, output_path)
                    else:
                        pt_files = [f for f in os.listdir(results_foldertag) if f.endswith('.pt')]
                        if len(pt_files) < 1:
                            print(f"{results_foldertag} 中的.pt文件少于4个，重新训练。")
                            start_training(config, data_root, results_folder, sparsity, dataset_name, sparsity_percentage)
                            run_sampling(data_root,dataset_name, sparsity_percentage, results_folder, output_path)
                        else:
                            if not os.path.exists(output_path):
                                print(f"{output_path} 不存在，执行采样。")
                                run_sampling(data_root,dataset_name, sparsity_percentage, results_folder, output_path)
                            else:
                                print(f"{results_foldertag} 已存在，且 {output_path} 输出已存在，跳过。")

def start_training(config, data_root, results_folder, sparsity, dataset_name, sparsity_percentage):
    config['dataloader']['train_dataset']['params']['data_root'] = data_root
    config['dataloader']['train_dataset']['params']['proportion'] = sparsity
    config['solver']['results_folder'] = results_folder

    with open(config_file_path, 'w') as file:
        yaml.dump(config, file)

    train_model(dataset_name, sparsity_percentage)

def train_model(dataset_name, sparsity_percentage):
    name = f"{dataset_name}_{sparsity_percentage}"
    command = f"python main.py --name {name} --config_file {config_file_path} --gpu 0 --train"
    os.system(command)
    print(f"训练命令已执行：{command}")

def run_sampling(data_root,dataset_name, sparsity_percentage, results_folder, output_path):
    # 更新 sample.yaml 中的 results_folder
    with open(sample_config_path, 'r') as file:
        sample_config = yaml.safe_load(file)
    sample_config['dataloader']['train_dataset']['params']['data_root'] = data_root
    sample_config['solver']['results_folder'] = results_folder

    with open(sample_config_path, 'w') as file:
        yaml.dump(sample_config, file)

    os.makedirs(output_path, exist_ok=True)

    name = f"{dataset_name}_{sparsity_percentage}"
    checkpoint_number = 1  # 可根据需要调整
    command = f"python main.py --name {name} --config_file {sample_config_path} --gpu 0 --sample 0 --milestone {checkpoint_number} --output {output_path}"
    os.system(command)
    print(f"采样命令已执行：{command}")

if __name__ == '__main__':
    generate_csv_files()
