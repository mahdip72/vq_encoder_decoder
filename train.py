import argparse
import yaml
import torch
from utils import *
from accelerate import Accelerator
from data_test import *
from model import SimpleVQAutoEncoder
from tqdm import tqdm


def train_loop(model, train_loader, optimizer, epoch, alpha, num_codes):
    model.train()
    total_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Training Epoch {epoch}")
    for data in pbar:
        inputs, labels = data
        # inputs = inputs.to(device)
        optimizer.zero_grad()
        outputs, indices, cmt_loss = model(inputs)
        rec_loss = torch.abs(outputs - inputs).mean()
        loss = rec_loss + alpha * cmt_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        batch_avg_loss = total_loss / (pbar.n + 1)
        pbar.set_description(
            f"Epoch: {epoch}, Batch Avg Loss: {batch_avg_loss:.3f} | "
            + f"Rec Loss: {rec_loss.item():.3f} | "
            + f"Cmt Loss: {cmt_loss.item():.3f} | "
            + f"Active %: {indices.unique().numel() / num_codes * 100:.3f}"
        )

    avg_loss = total_loss / len(train_loader)
    return avg_loss


def main(dict_config, config_file_path):
    configs = load_configs(dict_config)

    if isinstance(configs.fix_seed, int):
        torch.manual_seed(configs.fix_seed)
        torch.random.manual_seed(configs.fix_seed)
        np.random.seed(configs.fix_seed)

    result_path, checkpoint_path = prepare_saving_dir(configs, config_file_path)

    logging = get_logging(result_path)

    alpha = 10
    num_codes = 256
    train_dataloader = load_fashion_mnist_data(batch_size=256, shuffle=True)

    accelerator = Accelerator(
        mixed_precision=configs.train_settings.mixed_precision,
        gradient_accumulation_steps=configs.train_settings.grad_accumulation,
        dispatch_batches=True
    )

    net = SimpleVQAutoEncoder(codebook_size=256)

    optimizer, scheduler = prepare_optimizer(net, configs, len(train_dataloader), logging)
    logging.info('preparing optimizer is done')

    net, optimizer, train_dataloader, scheduler = accelerator.prepare(
        net, optimizer, train_dataloader, scheduler
    )

    net.to(accelerator.device)

    # compile model to train faster and efficiently
    if configs.prot2seq_model.compile_model:
        net = torch.compile(net)
        if accelerator.is_main_process:
            logging.info('compile model is done')

    # Initialize train and valid TensorBoards
    train_writer, valid_writer = prepare_tensorboard(result_path)

    for epoch in range(1, configs.train_settings.num_epochs + 1):
        train_loss = train_loop(net, train_dataloader, optimizer, epoch, alpha, num_codes)
        print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}')

    print("Training complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a VQ-VAE model.")
    parser.add_argument("--config_path", "-c", help="The location of config file", default='./config.yaml')
    args = parser.parse_args()
    config_path = args.config_path

    with open(config_path) as file:
        config_file = yaml.full_load(file)

    main(config_file, config_path)
