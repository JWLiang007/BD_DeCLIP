# Standard Library
import logging
import os

# Import from third library
import torch
import torch.distributed as dist
import functools
import socket
from datetime import timedelta

logger = logging.getLogger('global')
# This file should never dependent on pod
# from pod.utils.log_helper import default_logger as logger
allreduce = dist.all_reduce
allgather = dist.all_gather
broadcast = dist.broadcast
synchronize = torch.cuda.synchronize
init_process_group = dist.init_process_group
allreduce_async = functools.partial(dist.all_reduce, async_op=True)



def get_rank():
    return int(os.environ.get('SLURM_PROCID', 0))


def get_world_size():
    return int(os.environ.get('SLURM_NTASKS', 1))


def barrier():
    if get_world_size() > 1:
        x = torch.cuda.IntTensor([1])
        dist.all_reduce(x)
        x.cpu()


def get_local_rank():
    rank = dist.get_rank()
    return rank % torch.cuda.device_count()

def get_node_list():
    return os.environ.get('SLURM_NODELIST','localhost.localdomain')

def check_port_in_use(port, host='127.0.0.1'):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((host, int(port)))
        s.settimeout(1)
        s.shutdown(2)
        return True
    except:
        return False

def initialize(backend='nccl'):
    port = "12345"
    proc_id = int(get_rank())
    ntasks = int(get_world_size())
    node_list = str(get_node_list())
    if '[' in node_list:
        beg = node_list.find('[')
        pos1 = node_list.find('-', beg)
        if pos1 < 0:
            pos1 = 1000
        pos2 = node_list.find(',', beg)
        if pos2 < 0:
            pos2 = 1000
        node_list = node_list[:min(pos1, pos2)].replace('[', '')
    # addr = node_list[8:].replace('-', '.')
    addr = node_list
    os.environ['MASTER_PORT'] = port
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['RANK'] = str(proc_id)
    if backend == 'nccl':
        while True:
            try:
                dist.init_process_group(backend='nccl',timeout=timedelta(seconds=5))
                break 
            except Exception:
                os.environ['MASTER_PORT'] = str(int(port)+1)
    else:
        dist.init_process_group(backend='gloo', rank=proc_id, world_size=ntasks)
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)


def finalize():
    pass


# class nn(object):
#     SyncBatchNorm2d = torch.nn.BatchNorm2d
#     logger.info("You are using fake SyncBatchNorm2d who is actually the official BatchNorm2d")


# class syncbnVarMode_t(object):
#     L1 = None
#     L2 = None


# class optim(object):

#     class FusedFP16SGD(object):

#         def __init__(self, *args, **kwargs):
#             pass

#         @property
#         def optimizer(self):
#             return self
