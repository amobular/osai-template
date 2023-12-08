import zipfile
from pathlib import Path

import hydra

from src.conf import PrepareDataConfig


def download_data(cfg: PrepareDataConfig):
    if not cfg.download:
        return

    try:
        import kaggle
    except ImportError:
        raise RuntimeError(
            f"Could not import kaggle, make sure you have it installed and have a .json in your .kaggle "
            f"folder. Otherwise, if you are running on Kaggle, use the 'kaggle_overrides' parameter."
        )

    print("Downloading dataset...")
    data_dir = Path(cfg.dir.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    kaggle.api.authenticate()
    kaggle.api.competition_download_files(cfg.competition_name, path=data_dir, quiet=False)

    print("Unzipping downloaded dataset...")
    with zipfile.ZipFile(data_dir / f"{cfg.competition_name}.zip", "r") as zip_ref:
        zip_ref.extractall(data_dir)


def prepare_data(cfg: PrepareDataConfig):
    """
    This function actually prepares the data and outputs to cfg.dir.processed_dir.
    """
    processed_dir = Path(cfg.dir.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    pass


@hydra.main(config_path="conf", config_name="prepare_data", version_base="1.2")
def main(cfg: PrepareDataConfig):
    download_data(cfg)
    prepare_data(cfg)


if __name__ == '__main__':
    main()
