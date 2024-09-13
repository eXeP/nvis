
import os
import inspect
import torch
import torchvision
import json
import subprocess
import random

from pathlib import Path


# TODO add support for np array here (np.array) -> torch.tensor
# TODO add support for pil image here
# TODO add support for hist using matplotlib
# TODO add exr-support for hdr-valued
# TODO add whot() like support


def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_back.f_locals.items()
    names = [var_name for var_name, var_val in callers_local_vars if var_val is var]
    if len(names) == 0:
        return None
    else:
        return names[0]

def save_single(tensor, name, vis_dir):
    save_path = vis_dir / name
    torchvision.utils.save_image(tensor, save_path)
    return name


global server_processes
server_processes = []

def tensor_to_n3hw(tensor):
    if len(tensor.shape) == 3:
            tensor = tensor.unsqueeze(0)
    elif len(tensor.shape) == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
    elif len(tensor.shape) != 4:
        return None

    if tensor.shape[1] == 1:
        tensor = tensor.repeat(1, 3, 1, 1)
    elif tensor.shape[1] == 2:
        tensor = torch.cat((tensor, torch.zeros_like(tensor)[:, 0:1, ...]), dim=1)
    elif tensor.shape[1] > 3:
        tensor = tensor[:, 0:3, ...]
    return tensor

def start_http_server(nvis_path):
    kill_servers()
    nvis_server_command = ['python', '-m', 'http.server', '--bind', 'localhost', '8088', '-d', nvis_path]
    server_processes.append(
        subprocess.Popen(nvis_server_command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    )


def plot(*lists):
    nvis_path = Path(os.path.dirname(os.path.realpath(__file__)))
    (nvis_path / 'vis').mkdir(exist_ok=True)

    # Imports here so plt is not needed for torch plotting
    import matplotlib.pyplot as plt
    import numpy as np
    from typing import List

    plt.clf()
    streams = []
    nameless_tensor_idx = 0
    for list in lists:
        list_name = retrieve_name(list)
        if list_name == None:
            list_name = "VISDBG_{:2d}".format(nameless_tensor_idx)
            nameless_tensor_idx += 1
        if type(list) == List:
            list = np.array(list)
        plt.plot(np.arange(len(list)), list, '--o', label=list_name)
    plt.legend(loc="best")
    plt.savefig(nvis_path / 'vis/plot.png')
    
    stream = {
        'name': 'plot',
        'window': True,
        'images': ['vis/plot.png']
    }
    streams.append(stream)
    nvis_config = {
            'name': 'vis-script for plotting',
            'streams': streams,
        }
    with open(Path(nvis_path) / 'vis/nvis_config.json', 'w') as f:
        f.write(json.dumps(nvis_config))
    
    start_http_server(nvis_path)

def vis(*tensors, normalize=None, bidx=0, append=False, name=None):
    """
    Writes tensors to disk as nvis streams and starts a http server to display them
    """
    nvis_path = Path(os.path.dirname(os.path.realpath(__file__)))
    (nvis_path / 'vis').mkdir(exist_ok=True)

    streams = []
    nameless_tensor_idx = 0
    if append == True and (nvis_path /'vis/nvis_config.json').exists():
        prev_config = json.load(open(Path(nvis_path) / 'vis/nvis_config.json', 'r'))
        for stream in prev_config['streams']:
            if 'VISDBG_' in stream['name']:
                prev_nameless_idx = int(stream['name'].replace('VISDBG_', ''))
                nameless_tensor_idx = max(nameless_tensor_idx, prev_nameless_idx+1)
        
    for tensor in tensors:
        if len(tensors) == 1 and name != None:
            tensor_name = name
        else:
            tensor_name = retrieve_name(tensor)
        if tensor_name == None:
            tensor_name = "VISDBG_{:02d}".format(nameless_tensor_idx)
            nameless_tensor_idx += 1

        if not normalize:
            t_min = tensor.min()
            t_max = tensor.max()
            if t_min >= 0.0 and t_max <= 1.0 or tensor.dtype == torch.uint8:
                normalize = lambda x: x
            elif t_min >= -1.4 and t_max <= 1.4: # [-1, 1] with overshoots
                normalize = lambda x: x.add(1).div(2).clamp(0, 1)
            else:
                normalize = lambda x: x.sigmoid()
            
        images = []
            
        tensor = tensor_to_n3hw(tensor)
        if tensor is None:
            continue
        
        bs = tensor.shape[0]
        if bidx == None:
            vis_range = range(bs)
        else:
            vis_range = [bidx]
        tensor_id = random.randint(0, 10000000000)
        for bs_i in vis_range:
            rel_path = save_single(normalize(tensor[bs_i:bs_i+1]), 'vis/{}_{:012d}_{:04d}.png'.format(tensor_name, tensor_id,  bs_i), nvis_path)
            images.append(str(rel_path))
        
        stream = {
            'name': tensor_name,
            'window': True,
            'images': images
        }
        streams.append(stream)
    if append:
        nvis_config = json.load(open(Path(nvis_path) / 'vis/nvis_config.json', 'r'))
        nvis_config['streams'].extend(streams)
    else:
        nvis_config = {
            'name': 'vis-script for pytorch/np/PIL visualization',
            'streams': streams,
        }
    with open(Path(nvis_path) / 'vis/nvis_config.json', 'w') as f:
        f.write(json.dumps(nvis_config))
    
    start_http_server(nvis_path)

def kill_servers():
    global server_processes
    for server_process in server_processes:
        server_process.kill()
        server_processes.remove(server_process)
