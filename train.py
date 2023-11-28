import gc

import torch
import wandb
from absl import app, flags
from ml_collections import config_flags
from torch.optim import AdamW

from hivegraph.engine import Trainer
from hivegraph.io.autodataset import AutoDataset
from hivegraph.nn.automodel import AutoModel
from hivegraph.utils import set_seed

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    set_seed(FLAGS.config.random_seed)

    if FLAGS.config.log_to_wandb:
        assert (
            FLAGS.config.wandb_project is not None
            and FLAGS.config.wandb_entity is not None
        ), "Please provide valid wandb project and entity names."
        wandb.init(
            project=FLAGS.config.wandb_project,
            entity=FLAGS.config.wandb_entity,
            config=FLAGS.config.to_dict(),
        )

    dataset = AutoDataset(dataset_name=FLAGS.config.dataset)
    dataset = dataset.get_dataset()

    gnn = AutoModel(
        num_features=dataset.num_features,
        num_classes=dataset.num_classes,
        model_config=FLAGS.config.model,
    ).to(device)

    trainer = Trainer(
        model=gnn,
        dataset=dataset,
        device=device,
        criterion=FLAGS.config.criterion,
        num_folds=FLAGS.config.num_folds,
        random_state=FLAGS.config.random_seed,
        test_metric=FLAGS.config.test_metric,
    )

    trainer.fit(
        epochs=FLAGS.config.num_epochs,
        batch_size=FLAGS.config.batch_size,
        optimizer=AdamW(gnn.parameters(), lr=FLAGS.config.lr),
        verbose=FLAGS.config.verbose,
        log_to_wandb=FLAGS.config.log_to_wandb,
    )

    if FLAGS.config.log_to_wandb:
        wandb.finish()

    _ = gc.collect()


if __name__ == "__main__":
    flags.mark_flags_as_required(["config"])
    app.run(main)
