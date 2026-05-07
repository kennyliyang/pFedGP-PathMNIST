Personalized Federated Learning with Gaussian Processes (PathMNIST Edition)
Federated learning aims to learn a global model that performs well on client devices with limited cross-client communication. Personalized federated learning (PFL) further extends this setup to handle data heterogeneity between clients by learning personalized models. A key challenge in this setting is to learn effectively across clients, even though each client has its unique data, which is often limited in size.  

Here we put forward a solution to PFL that is based on Gaussian processes (GPs) with deep kernel learning, which we call pFedGP. In this extended project, we specifically adapt and deeply optimize the pFedGP framework for medical pathology image classification using the PathMNIST (MedMNIST) dataset. We provide a unified pipeline to compare pFedGP against classic baselines (FedAvg, FedPer) with automatic dataset inference, and ensure seamless deployment on modern NVIDIA RTX 50-series GPUs (CUDA 12.1+). We find that pFedGP significantly outperforms baseline methods while achieving well-calibrated predictions in the medical domain.  

[Original Paper]
[Original Project-Page]

Instructions
1. Install Environment (Optimized for RTX 50-series):

Bash
conda create -n pfedgp_5080 python=3.10 -y
conda activate pfedgp_5080

# Install PyTorch for CUDA 12.1+
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install dataset tools and repo dependencies
pip install medmnist tqdm numpy pandas matplotlib
pip install -e .
2. Download and Prepare Data:
To generate the PathMNIST dataset in the required 100-client dictionary format:

Bash
cd experiments/datafolder/
python generate_medmnist_pkl.py
(Optional) To download noisy CIFAR-10 or CIFAR-100:

Bash
cd experiments/datafolder/noisy_cifar10
python download_noisy_data.py
3. Run Experiments:
All training scripts are unified under the experiments/noisy_input directory. The codebase features automatic class inference based on the --data-name (e.g., pathmnist: 9, cifar10: 10).

To run pFedGP variants (pFedGP-compute or pFedGP-data):

Bash
cd experiments/noisy_input/
python trainer_ip.py \
    --method pFedGP-compute \
    --data-name pathmnist \
    --data-path ../datafolder/medmnist_path_dictionary.pkl \
    --num-steps 500
To run Baselines (fedavg or fedper):

Bash
cd experiments/noisy_input/
python trainer_baselines.py \
    --method fedper \
    --data-name pathmnist \
    --data-path ../datafolder/medmnist_path_dictionary.pkl \
    --num-steps 500
Note: You can easily run experiments on other datasets by replacing --data-name pathmnist with cifar10 or cifar100, and adjusting the --data-path accordingly.

Citation
Please cite the original paper if you want to use the core pFedGP framework in your work:

Code snippet
@article{achituve2021personalized,
  title={Personalized Federated Learning with Gaussian Processes},
  author={Achituve, Idan and Shamsian, Aviv and Navon, Aviv and Chechik, Gal and Fetaya, Ethan},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}

