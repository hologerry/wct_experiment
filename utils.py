"""Util functions
"""
import json
from datetime import datetime

import torch


def write_event(log, step: int, **data):
    data['step'] = step
    data['time'] = datetime.now().isoformat()
    log.write(json.dumps(data, sort_keys=True))
    log.write('\n')
    log.flush()


def save_model(decoder, relu_target_id, epoch, step, model_path):
    torch.save({
        'model': decoder.state_dict(),
        'relu_target': relu_target_id,
        'epoch': epoch,
        'step': step,
    }, str(model_path))
