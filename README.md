Human Activity Recognition Experiment (Folder name: Human_Activity_Recognition):

Step 1: Modify the following three paths to the location where the Human_Activity_Recognition folder is stored:

In the config folder, modify line 22 of config.gin: load.data_dir = "/home/RUS_CIP/st191716/dl-lab-24w-team08/Human_Activity_Recognition"

In the input_pipeline folder, modify line 272 of datasets.py：output_path = "/home/RUS_CIP/st191716/dl-lab-24w-team08/Human_Activity_Recognition"

In main.py, modify line 28:data_dir = "/home/RUS_CIP/st191716/dl-lab-24w-team08/Human_Activity_Recognition"

Step 2: Run the following script in train.sh:

#!/bin/bash -l

#Slurm parameters

#SBATCH --job-name=job_name

#SBATCH --output=job_name-%j.%N.out

#SBATCH --time=1-00:00:00

#SBATCH --gpus=1

#Activate everything you need

module load cuda/11.8

#Run your python code

python input_pipeline/datasets.py

python main.py --train=True

This script processes the dataset and generates TFRecord files, which are stored in the Human_Activity_Recognition folder. Afterward, the program trains the model, and the training results are saved in the experiments2 folder.

Step 3: Run the following script in train.sh to evaluate the model:

#!/bin/bash -l

#Slurm parameters

#SBATCH --job-name=job_name

#SBATCH --output=job_name-%j.%N.out

#SBATCH --time=1-00:00:00

#SBATCH --gpus=1

#Activate everything you need

module load cuda/11.8

#Run your python code

python main.py --train=False

This script evaluates the trained model, and the evaluation results are saved in the experiments2 folder.

Optional Step: Run the following script in train.sh for hyperparameter optimization:

#!/bin/bash -l

#Slurm parameters

#SBATCH --job-name=job_name

#SBATCH --output=job_name-%j.%N.out

#SBATCH --time=1-00:00:00

#SBATCH --gpus=1

#Activate everything you need

module load cuda/11.8

#Run your python code

python grid.py

This script performs hyperparameter optimization.
