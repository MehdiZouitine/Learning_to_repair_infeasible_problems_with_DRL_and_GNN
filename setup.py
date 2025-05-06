from setuptools import setup, find_packages

setup(
    name="LearningToRepairDRL",  # Name of your package
    version="0.1.0",
    description="Learning to repair infeasible problems with DRL and GNN",
    author="Mehdi Zouitine",
    author_email="mehdizouitinegm@gmail.com",
    packages=find_packages(),
    install_requires=[
        "gymnasium",
        "numpy",
        "torch",
        "tyro",
        "tqdm",
        "tensorboard",
        "wandb",
        "python-sat",
        "einops",
        "torch_geometric",
        "scipy",
    ],
    python_requires=">=3.8",
)