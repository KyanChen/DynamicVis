# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import sys
import time

import torch
import tqdm

sys.path.append(sys.path[0] + '/../..')
from mmengine.analysis import get_model_complexity_info

from mmpretrain import get_model


def parse_args():
    parser = argparse.ArgumentParser(description='Get model flops and params')
    parser.add_argument('--config', default='configs_DynamicVis/fMoW/pretrain_dynamicvis_b_bf16_mamba.py', help='config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[512],
        help='input image size')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3, ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    model = get_model(args.config)
    model.eval()
    model.to('cuda')

    if hasattr(model, 'extract_feat'):
        model.forward = model.extract_feat
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
            format(model.__class__.__name__))

    # 准备数据
    batch_size = 1
    # find the biggest batch size that can run on the device

    torch.cuda.empty_cache()
    initial_mem = torch.cuda.memory_allocated()

    input_data = torch.randn(1, *input_shape)
    input_data = input_data.to('cuda')
    with torch.no_grad():
        _ = model(input_data)
        peak_mem = torch.cuda.max_memory_allocated()

    # 寻找batch size
    while True:
        try:
            input_data = torch.randn(batch_size, *input_shape)
            input_data = input_data.to('cuda')
            with torch.no_grad():
                _ = model(input_data)
            torch.cuda.empty_cache()
            break
        except RuntimeError as e:
            batch_size = batch_size - 4
            torch.cuda.empty_cache()
            print(f'{e}: {batch_size}')
    torch.cuda.empty_cache()
    print(f'batch_size: {batch_size}', 'finished')

    # 将模型和数据移动到GPU（如果可用）
    input_data = torch.randn(batch_size, *input_shape).to('cuda')
    # 预热模型（避免第一次运行时的额外开销）
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_data)

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    # 计时开始
    i_ter_steps = 20
    total_time = 0

    with torch.no_grad():
        for _ in tqdm.tqdm(range(i_ter_steps)):
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
            _ = model(input_data)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender) / 1000
            total_time += curr_time

    # 5. 计算显存占用（包括模型参数 + 输入/输出 + 中间变量）
    print(f"模型加载后基础显存: {initial_mem / 1024 ** 2:.2f} MB")
    print(f"推理峰值显存: {peak_mem / 1024 ** 2:.2f} MB")
    print(f"推理额外占用: {(peak_mem - initial_mem) / 1024 ** 2:.2f} MB")

    throughput = batch_size * i_ter_steps / total_time
    print(f'Throughput: {throughput:.2f} samples/s')
    print(f'Latency: {1000 * total_time / (batch_size * i_ter_steps):.2f} ms')

if __name__ == '__main__':
    main()
