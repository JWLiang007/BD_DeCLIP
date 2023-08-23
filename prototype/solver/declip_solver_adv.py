import os
import argparse
from easydict import EasyDict
from tensorboardX import SummaryWriter
import pprint
import time
import datetime
import torch
import json
import math
import linklink as link
import torch.nn.functional as F

from .base_solver import BaseSolver
from prototype.utils.dist import link_dist, DistModule, broadcast_object
from prototype.utils.misc import makedir, create_logger, get_logger, count_params, count_flops, \
    param_group_all, AverageMeter, accuracy, load_state_model, load_state_optimizer, mixup_data, \
    mix_criterion, modify_state, cutmix_data, parse_config
from prototype.model.image_encoder.modified_resnet import modified_resnet_R50
from prototype.backdoor.at import AT
from prototype.utils.ema import EMA
from prototype.utils.misc import visual_img,save_imgs
from prototype.model import model_entry
from prototype.optimizer import optim_entry, FP16RMSprop, FP16SGD, FusedFP16SGD, FP16AdamW
from prototype.lr_scheduler import scheduler_entry
from prototype.data import build_imagenet_train_dataloader, build_imagenet_test_dataloader
from prototype.data import build_clip_dataloader
from prototype.loss_functions import LabelSmoothCELoss, ClipInfoCELoss, SimsiamLoss, NTXentLoss
# from prototype.utils.user_analysis_helper import send_info
from prototype.utils.grad_clip import clip_grad_norm_, clip_grad_value_, clip_param_grad_value_
from tqdm import tqdm

class DataPrefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return
        # with torch.cuda.stream(self.stream):
        #     for k in self.batch:
        #         if k != 'meta':
        #             self.batch[k] = self.batch[k].to(device=self.opt.device, non_blocking=True)

            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            #     self.next_input = self.next_input.float()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch


class EMA_logit_scale():
    def __init__(self, param, threshold):
        self.param = param
        self.buffer = 3.125
        self.momentum = 0.9
        self.threshold = threshold
        self.clip_number = 0

    def update(self):
        self.buffer = self.momentum*self.buffer + \
            (1-self.momentum)*self.param.data

    def clamp(self):
        if (self.param-self.buffer) > self.threshold:
            self.param.data = torch.as_tensor(
                self.buffer+self.threshold, dtype=self.param.dtype, device=self.param.device)
            self.clip_number += 1
        elif (self.buffer-self.param) > self.threshold:
            self.param.data = torch.as_tensor(
                self.buffer-self.threshold, dtype=self.param.dtype, device=self.param.device)
            self.clip_number += 1
        # self.param.data = torch.as_tensor(
        #     3.125, dtype=self.param.dtype, device=self.param.device)


class ClsSolver(BaseSolver):

    def __init__(self, config_file):
        self.config_file = config_file
        self.prototype_info = EasyDict()
        self.config = parse_config(config_file)
        self.setup_env()
        # import ipdb
        # ipdb.set_trace()
        self.build_tea()
        self.build_model()
        self.build_optimizer()
        self.build_data()
        self.build_lr_scheduler()
        # send_info(self.prototype_info)
        
    def build_tea(self):
        tea_config = self.config.model.kwargs
        self.tea_image_encode=None
        if  tea_config.get('defense',None) is not None :
            tea_image_encode = modified_resnet_R50(**tea_config['image_encode'])
            tea_weight = tea_config['defense']['ft_weight_path']
            loaded_dict = torch.load(tea_weight, 'cpu')
            tea_dict = {}
            for k,v in loaded_dict['model'].items():
                if 'visual' in k:
                    tea_dict[k.replace('module.visual.','')] = v
            tea_image_encode.load_state_dict(tea_dict, strict=True)
            tea_image_encode.eval()
            self.tea_image_encode = tea_image_encode
            self.tea_image_encode.cuda()
            self.at = AT(2)
            self.tea_image_encode = DistModule(self.tea_image_encode, self.config.dist.sync)
    def setup_env(self):
        # dist
        self.dist = EasyDict()
        self.dist.rank, self.dist.world_size = link.get_rank(), link.get_world_size()
        self.prototype_info.world_size = self.dist.world_size
        # directories
        self.path = EasyDict()
        self.path.root_path = os.path.dirname(self.config_file)
        self.path.save_path = os.path.join(self.path.root_path, 'checkpoints',self.config.saver.save_prefix)
        self.path.event_path = os.path.join(self.path.root_path, 'events')
        self.path.result_path = os.path.join(self.path.root_path, 'results')
        makedir(self.path.save_path)
        makedir(self.path.event_path)
        makedir(self.path.result_path)
        # tb_logger
        if self.dist.rank == 0:
            self.tb_logger = SummaryWriter(self.path.event_path)
        # logger
        create_logger(os.path.join(self.path.root_path, 'log.txt'))
        self.logger = get_logger(__name__)
        self.logger.critical(f'config: {pprint.pformat(self.config)}')
        if 'SLURM_NODELIST' in os.environ:
            self.logger.critical(f"hostnames: {os.environ['SLURM_NODELIST']}")
        # load pretrain checkpoint
        if hasattr(self.config.saver, 'pretrain'):
            if not self.config.saver.pretrain.get("path", None):
                if self.config.saver.pretrain.auto_resume:
                    last_checkpoint = self.find_last_checkpoint()
                    if last_checkpoint:
                        self.config.saver.pretrain.path = last_checkpoint
            if self.config.saver.pretrain.get("path", None):
                self.state = torch.load(self.config.saver.pretrain.path, 'cpu')
                # for key in ['optimizer','last_iter' ]:
                #     if key in self.state:
                #         self.state.pop(key)
                self.logger.info(
                    f"Recovering from {self.config.saver.pretrain.path}, keys={list(self.state.keys())}")
            else:
                self.state = {}
                self.state['last_iter'] = 0
            #pretrain from moco
            if self.config.saver.pretrain.get('pretrain_from', None)  == 'moco':
                encoder_state = {}
                for key, value in self.state['model'].items():
                    if 'encoder_q' in key and 'fc' not in key and 'attnpool' not in key:
                        new_key = key.replace('encoder_q', 'visual')
                        encoder_state[new_key] = value
                self.state = {'model': encoder_state, 'last_iter': 0, 'optimizer': None}
            #pretrain from supervised
            if self.config.saver.pretrain.get('pretrain_from', None)  == 'supervised':
                encoder_state = {}
                for key, value in self.state['model'].items():
                    if 'fc' not in key:
                        new_key = key.replace('module', 'module.visual')
                        encoder_state[new_key] = value
                self.state = {'model': encoder_state, 'last_iter': 0, 'optimizer': None}

            if hasattr(self.config.saver.pretrain, 'ignore'):
                self.state = modify_state(
                    self.state, self.config.saver.pretrain.ignore)

        else:
            self.state = {}
            self.state['last_iter'] = 0
        # others
        torch.backends.cudnn.benchmark = True

    def find_last_checkpoint(self):
        ckpt_list = os.listdir('checkpoints')
        if 'ckpt.pth.tar' in ckpt_list:
            return 'checkpoints/ckpt.pth.tar'
        elif len(ckpt_list) == 0:
            return None
        num = [int(ckpt.split('.')[0][5:]) for ckpt in ckpt_list]
        num.sort()
        last_checkpoint_path = 'checkpoints/ckpt_' + str(num[-1])+'.pth.tar'
        return last_checkpoint_path

    def build_model(self):
        if hasattr(self.config, 'lms'):
            if self.config.lms.enable:
                torch.cuda.set_enabled_lms(True)
                byte_limit = self.config.lms.kwargs.limit * (1 << 30)
                torch.cuda.set_limit_lms(byte_limit)
                self.logger.critical('Enable large model support, limit of {}G!'.format(
                    self.config.lms.kwargs.limit))

        self.model = model_entry(self.config.model)
        self.prototype_info.model = self.config.model.type
        self.model.cuda()

        count_params(self.model)
        # count_flops(self.model, input_shape=[
        #             1, 3, self.config.data.input_size, self.config.data.input_size])

        # handle fp16
        if self.config.optimizer.type == 'FP16SGD' or \
           self.config.optimizer.type == 'FP16RMSprop' or \
           self.config.optimizer.type == 'FP16AdamW'or \
           self.config.optimizer.type == 'FP16AdamW_SGD':
            self.fp16 = True
            self.fused_fp16 = False
        elif self.config.optimizer.type == 'FusedFP16SGD' or \
             self.config.optimizer.type == 'FusedFP16AdamW':
            self.fp16 = True
            self.fused_fp16 = True

        else:
            self.fp16 = False
            self.fused_fp16 = False

        if self.fp16:
            # if you have modules that must use fp32 parameters, and need fp32 input
            # try use link.fp16.register_float_module(your_module)
            # if you only need fp32 parameters set cast_args=False when call this
            # function, then call link.fp16.init() before call model.half()
            self.logger.critical('using fp16 when training')
            if self.config.optimizer.get('fp16_normal_bn', False):
                self.logger.critical('using normal bn for fp16')
                link.fp16.register_float_module(
                    link.nn.SyncBatchNorm2d, cast_args=False)
                link.fp16.register_float_module(
                    torch.nn.BatchNorm2d, cast_args=False)
            if self.config.optimizer.get('fp16_normal_fc', False):
                self.logger.critical('using normal fc for fp16')
                link.fp16.register_float_module(
                    torch.nn.Linear, cast_args=True)
            if self.config.optimizer.get('fp16_normal_ln', False):
                self.logger.critical('using normal ln for fp16')
                link.fp16.register_float_module(
                    torch.nn.LayerNorm, cast_args=False)
            link.fp16.init()
            # Note: The module is converted to fp16!
            self.model.half()
        self.model = DistModule(self.model, self.config.dist.sync)

        if 'model' in self.state:
            load_state_model(self.model, self.state['model'])

    def build_optimizer(self):

        opt_config = self.config.optimizer
        opt_config.kwargs.lr = self.config.lr_scheduler.kwargs.base_lr
        self.prototype_info.optimizer = self.config.optimizer.type

        # make param_groups
        pconfig = {}

        if opt_config.get('no_wd', False):
            pconfig['conv_b'] = {'weight_decay': 0.0}
            pconfig['linear_b'] = {'weight_decay': 0.0}
            pconfig['bn_w'] = {'weight_decay': 0.0}
            pconfig['bn_b'] = {'weight_decay': 0.0}

        if 'pconfig' in opt_config:
            pconfig.update(opt_config['pconfig'])

        #if opt_config['type'] == 'AdamW_SGD':
        if opt_config['type'] in ['FP16AdamW_SGD', 'AdamW_SGD']:
            text_config = opt_config['text_config']
            visual_config = opt_config['visual_config']
            text_parameters = self.model.module.text_parameters()
            visual_parameters = self.model.module.visual_parameters()
            param_group = []
            if len(text_parameters) > 0:
                param_group.append({'params': text_parameters, **text_config})
            if len(visual_parameters) > 0:
                param_group.append(
                    {'params': visual_parameters, **visual_config})
            #for text_module in self.model.module.text_modules():
            self.logger.critical('[Info] param group: Text-module')
            for idx, text_module in enumerate(self.model.module.text_modules()):
                self.logger.info(f'[Info] text param group {idx}')
                param_group_text, type2num = param_group_all(
                    text_module, pconfig, text_config)
                param_group += param_group_text
            #for visual_module in self.model.module.visual_modules():
            self.logger.critical('[Info] param group: Visual-module')
            for idx, visual_module in enumerate(self.model.module.visual_modules()):
                self.logger.info(f'[Info] visual param group {idx}')
                param_group_visual, type2num = param_group_all(
                    visual_module, pconfig, visual_config)
                param_group += param_group_visual
        else:
            param_group, type2num = param_group_all(self.model, pconfig)

        opt_config.kwargs.params = param_group

        self.optimizer = optim_entry(opt_config)

        if 'optimizer' in self.state:
            load_state_optimizer(self.optimizer, self.state['optimizer'])

        # EMA
        if self.config.ema.enable:
            self.config.ema.kwargs.model = self.model
            self.ema = EMA(**self.config.ema.kwargs)
        else:
            self.ema = None

        if 'ema' in self.state and self.config.ema.enable:
            self.ema.load_state_dict(self.state['ema'])

    def build_lr_scheduler(self):
        self.prototype_info.lr_scheduler = self.config.lr_scheduler.type
        if not getattr(self.config.lr_scheduler.kwargs, 'max_iter', False):
            self.config.lr_scheduler.kwargs.max_iter = self.config.data.max_iter
        self.config.lr_scheduler.kwargs.optimizer = self.optimizer.optimizer if isinstance(self.optimizer, FP16SGD) or \
            isinstance(self.optimizer, FP16RMSprop) or isinstance(self.optimizer, FP16AdamW) else self.optimizer
        self.config.lr_scheduler.kwargs.last_iter = self.state.get('last_iter',0)
        self.early_stop_iter = self.config.lr_scheduler.kwargs.pop('early_stop_iter',self.config.lr_scheduler.kwargs.max_iter)
        self.lr_scheduler = scheduler_entry(self.config.lr_scheduler)

    def build_data(self):
        test_config = {}
        self.config.data.last_iter = self.state.get('last_iter',0)
        test_config['last_iter'] = self.state.get('last_iter',0)
        if getattr(self.config.lr_scheduler.kwargs, 'max_iter', False):
            self.config.data.max_iter = self.config.lr_scheduler.kwargs.max_iter
            test_config['max_iter'] = self.config.lr_scheduler.kwargs.max_iter
        else:
            self.config.data.max_epoch = self.config.lr_scheduler.kwargs.max_epoch
            test_config['max_epoch'] = self.config.lr_scheduler.kwargs.max_epoch

        if self.config.data.get('type', 'imagenet') == 'imagenet':
            self.train_data = build_imagenet_train_dataloader(self.config.data)
        elif self.config.data.get('type') == 'clip':
            self.train_data = build_clip_dataloader('train', self.config.data)

        if self.config.data.get('type', 'imagenet') == 'imagenet':
            self.val_data = build_imagenet_test_dataloader(self.config.data)
        elif self.config.data.get('type') == 'clip':
            self.val_data = []
            for config in self.config.data.test:
                config.update(test_config)
                self.val_data.append(build_clip_dataloader('test', config))
                # _val_data = build_clip_dataloader('test', config)
                # _val_data_prefetcher = DataPrefetcher(_val_data['loader'])
                # _val_data['prefetcher'] = _val_data_prefetcher
                # self.val_data.append(_val_data)

        self.prefetch = self.config.data.train.get('prefetch', False)
        if self.prefetch:
            self.prefetcher = DataPrefetcher(self.train_data['loader'])
        elif self.train_data['loader']:
            self.train_data['loader'] = iter(self.train_data['loader'])

    def pre_train(self):
        self.meters = EasyDict()
        self.meters.batch_time = AverageMeter(self.config.saver.print_freq)
        self.meters.step_time = AverageMeter(self.config.saver.print_freq)
        self.meters.data_time = AverageMeter(self.config.saver.print_freq)
        self.meters.losses = AverageMeter(self.config.saver.print_freq)
        self.meters.nad_losses = AverageMeter(self.config.saver.print_freq)
        self.meters.clip_losses = AverageMeter(self.config.saver.print_freq)                         
        self.meters.simsiam_losses = AverageMeter(self.config.saver.print_freq)
        self.meters.text_simsiam_losses = AverageMeter(self.config.saver.print_freq)
        self.meters.clip_nn_text_losses = AverageMeter(self.config.saver.print_freq)
        self.meters.text_mlm_losses = AverageMeter(self.config.saver.print_freq)
        self.meters.nt_xent_losses = AverageMeter(self.config.saver.print_freq)
        self.meters.top1 = AverageMeter(self.config.saver.print_freq)
        self.meters.top5 = AverageMeter(self.config.saver.print_freq)

        self.model.train()

        label_smooth = self.config.get('label_smooth', 0.0)
        self.num_classes = self.config.model.kwargs.get('num_classes', 1000)
        self.topk = 5 if self.num_classes >= 5 else self.num_classes
        if label_smooth > 0:
            self.logger.critical('using label_smooth: {}'.format(label_smooth))
            self.criterion = LabelSmoothCELoss(label_smooth, self.num_classes)
        else:
            # self.criterion = ClipInfoCELoss(self.config.loss.partition_num)
            self.criterion = ClipInfoCELoss()
        self.simsiam_criterion = SimsiamLoss()
        self.nt_xent_criterion = NTXentLoss(self.config.data.batch_size)

        self.mixup = self.config.get('mixup', 1.0)
        self.cutmix = self.config.get('cutmix', 0.0)
        if self.mixup < 1.0:
            self.logger.critical(
                'using mixup with alpha of: {}'.format(self.mixup))
        if self.cutmix > 0.0:
            self.logger.critical(
                'using cutmix with alpha of: {}'.format(self.cutmix))
            

    
    def train(self):

        self.pre_train()
        total_step = len(self.train_data['loader'])
        start_step = self.state.get('last_iter',128000) + 1
        end = time.time()
        last_logit_scale = 0
        logit_scale = EMA_logit_scale(self.model.module.logit_scale,
                          self.config.grad_clip.value)
        #test 
        train_loss = 1000.0
        #test_top1_prec = 0.0
        #for id, val_data in enumerate(self.val_data):
        #    metrics = self.evaluate(val_data)
        #    test_top1_prec = metrics.metric['top1']

        _alpha =  0.05 
        _eps = 0.0125
        _decay = 0.9
        # import ipdb;ipdb.set_trace()
        for i in tqdm(range(len(self.train_data['loader']))):

            
            if self.prefetch:
                batch = self.prefetcher.next()
            else:
                batch = next(self.train_data['loader'])
            
            if i % 100 == 0:
                visual_img(batch,self.dist.rank,i, prefix='train')
                
            batch['images'] = batch['images'].cuda()
            
            images = batch['images'].detach().clone()
            ub = torch.max(images.view(*images.shape[:2],-1),dim=-1).values.unsqueeze(-1).unsqueeze(-1)
            lb = torch.min(images.view(*images.shape[:2],-1),dim=-1).values.unsqueeze(-1).unsqueeze(-1)
            eps = _eps * (ub - lb)
            alpha = _alpha * (ub - lb)
            momentum = torch.zeros_like(images)                
                
            for _iter in range(5):  
                curr_step = start_step + i
                       
                batch.images.requires_grad=True            
                if 'CLSA' not in self.config.data.train.transforms.type:
                    output_dict = self.model(batch, return_dict=True)

                    logits_per_image, logits_per_image_2, logits_per_text, logits_per_text_2 = output_dict['logits']
                    logits_per_image_1_aug, logits_per_image_2_aug, logits_per_text_1_aug, logits_per_text_2_aug = output_dict['logits_aug']
                    p1, p2, z1, z2 = output_dict['simsiam_features']
                    text_features, image_features_1, image_features_2 = output_dict['features']
                    # loss
                    clip_loss_1, target = self.criterion(logits_per_image, logits_per_text)
                    clip_loss_2, target_2 = self.criterion(logits_per_image_2, logits_per_text_2)
                    clip_loss_1_aug, target_1_aug = self.criterion(logits_per_image_1_aug, logits_per_text_1_aug)
                    clip_loss_2_aug, target_2_aug = self.criterion(logits_per_image_2_aug, logits_per_text_2_aug)

                    if self.config.data.train.get('only_image_two_view', False):
                        clip_loss = (clip_loss_1 + clip_loss_2) / 2
                    elif self.config.data.train.get('image_text_two_view', False):
                        clip_loss = (clip_loss_1 + clip_loss_2 + clip_loss_1_aug + clip_loss_2_aug) / 4
                    else:
                        clip_loss = clip_loss_1
                        # raise NotImplementedError()
                    clip_loss /= self.dist.world_size

                    if 'text_self_supervised' in output_dict.keys():
                        text_mlm_loss = output_dict['text_self_supervised'] / self.dist.world_size
                    else:
                        text_mlm_loss = torch.zeros_like(clip_loss)

                    if 'nn_text_simsiam' in output_dict.keys():
                        p_text, z_text_nn = output_dict['nn_text_simsiam']
                        z_text_nn = z_text_nn[0]
                        simsiam_nn_text_loss = self.simsiam_criterion(p_text,z_text_nn,p_text,z_text_nn, return_KPI=True)
                        simsiam_nn_text_loss = simsiam_nn_text_loss / self.dist.world_size  # z1d; z2d constructed
                    else:
                        simsiam_nn_text_loss = torch.zeros_like(clip_loss)

                    if 'text_simsiam' in output_dict.keys():
                        p1t, p2t, z1t, z2t = output_dict['text_simsiam']
                        text_simsiam_loss = self.simsiam_criterion(p1t,z1t,p2t,z2t) / self.dist.world_size
                    else:
                        text_simsiam_loss = torch.zeros_like(clip_loss)

                    if 'nn_text_logits' in output_dict.keys():
                        logits_per_image_1_nn, logits_per_image_2_nn, logits_per_image_1_nn_aug, logits_per_image_2_nn_aug = output_dict['nn_text_logits']
                        clip_loss_i1_nn, _ = self.criterion(logits_per_image_1_nn, logits_per_image_1_nn_aug)
                        clip_loss_i2_nn, _ = self.criterion(logits_per_image_2_nn, logits_per_image_2_nn_aug)
                        clip_nn_text_loss = (clip_loss_i1_nn+clip_loss_i2_nn)/2
                        clip_nn_text_loss = clip_nn_text_loss / self.dist.world_size
                        # print(clip_loss_i_nn, '<< clip loss i nn', flush=True)
                    else:
                        clip_nn_text_loss = torch.zeros_like(clip_loss)

                    simsiam_loss = self.simsiam_criterion(p1,z1,p2,z2) / self.dist.world_size

                    nt_xent_loss_1 = self.nt_xent_criterion(image_features_1, text_features)
                    nt_xent_loss_2 = self.nt_xent_criterion(image_features_2, text_features)
                    nt_xent_loss = (nt_xent_loss_1 + nt_xent_loss_2) / self.dist.world_size
                    # print(clip_loss, simsiam_loss, '<< losses', nt_xent_loss, flush=True)


                if not self.config.clip_simsiam_loss_weight.get('type', None):
                    loss = clip_loss * self.config.clip_simsiam_loss_weight.clip_loss
                    if self.config.clip_simsiam_loss_weight.get('simsiam_loss', 0):
                        loss = loss + simsiam_loss * self.config.clip_simsiam_loss_weight.simsiam_loss
                    if self.config.clip_simsiam_loss_weight.get('masking_language', 0):
                        loss = loss + text_mlm_loss * self.config.clip_simsiam_loss_weight.masking_language
                        # print('New Loss, text_mlm_loss', text_mlm_loss, flush=True)
                    if self.config.clip_simsiam_loss_weight.get('text_simsiam_loss', 0):
                        loss = loss + text_simsiam_loss * self.config.clip_simsiam_loss_weight.text_simsiam_loss
                    if self.config.clip_simsiam_loss_weight.get('nn_text', 0):
                        loss = loss + clip_nn_text_loss * self.config.clip_simsiam_loss_weight.nn_text
                        # print(clip_nn_text_loss, '<< nn text loss', flush=True)
                elif self.config.clip_simsiam_loss_weight.get('type', None) == 'convirt':
                    loss = (clip_loss + nt_xent_loss) / 2 * self.config.clip_simsiam_loss_weight.clip_loss + simsiam_loss * self.config.clip_simsiam_loss_weight.simsiam_loss
                elif self.config.clip_simsiam_loss_weight.get('type', None) == 'linear':
                    base_clip_loss_weight = 0.2
                    update_clip_loss_weight = 0.8 * curr_step / total_step
                    clip_loss_weight = base_clip_loss_weight + update_clip_loss_weight
                    simsiam_loss_weight = 1.0 - clip_loss_weight
                    loss = clip_loss  * clip_loss_weight + simsiam_loss * simsiam_loss_weight

                elif self.config.clip_simsiam_loss_weight.get('type', None) == 'shift':
                    if curr_step%2==0:
                        clip_loss_weight, simsiam_loss_weight = 1.0, 0.0
                    else:
                        clip_loss_weight, simsiam_loss_weight = 0.0, 1.0
                    loss = clip_loss * clip_loss_weight + simsiam_loss * simsiam_loss_weight

    
                # compute and update gradient
                self.optimizer.zero_grad()
                self.model.zero_grad()
                loss= loss* (-1.0)
                loss.backward()
                grad = batch.images.grad.data
                grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
                grad = grad + momentum * _decay
                momentum = grad

                batch.images = batch.images.detach() - alpha * grad.sign()
                delta = torch.clamp(batch.images - images, min=-eps, max=eps)
                batch.images = torch.clamp(images+ delta, min=lb, max=ub)

                self.logger.critical('iter: '+str(_iter) + '  loss: '+str(loss.item()))
            save_imgs(batch)
         

  
    
            
    @torch.no_grad()
    def evaluate(self, val_data):
        self.model.eval()
        res_file = os.path.join(self.path.result_path,
                                f'results.txt.rank{self.dist.rank}')
        writer = open(res_file, 'w')
        # label_ensemble
        # label_text, label_text_ensemble_matrix = val_data['loader'].dataset.get_label_texts(
        # )
        # label_text_preds = self.model.module.encode_text(label_text)
        # label_text_preds = label_text_preds / \
        #     (label_text_preds.norm(dim=-1, keepdim=True))

        label_text, label_text_ensemble_matrix = val_data['loader'].dataset.get_label_texts()
        label_num = label_text_ensemble_matrix.shape[1]
        prompts_num = len(label_text) // label_num
        self.logger.info('Use {} prompts'.format(prompts_num))
        label_text_preds = []
        for i in range(label_num):
            label_text_pred = self.model.module.encode_text(label_text[i*prompts_num:(i+1)*prompts_num])
            label_text_pred /= (label_text_pred.norm(dim=-1, keepdim=True))
            label_text_pred = label_text_pred.mean(dim=0)
            label_text_pred /= label_text_pred.norm()
            label_text_preds.append(label_text_pred)

        label_text_preds = torch.stack(label_text_preds, dim=0)

        label_text_ensemble_matrix = label_text_ensemble_matrix.to(
            label_text_preds)

        for batch_idx, batch in enumerate(tqdm(val_data['loader'])):
            input = batch['images']
            input = input.cuda().half() if self.fp16 else input.cuda()
            if batch_idx % (len(val_data['loader'])//5) == 0:
                visual_img(batch,self.dist.rank,batch_idx, prefix='test')
            # label = label.squeeze().view(-1).cuda().long()
            # compute output
            if self.config.get('return_dense', False):
                image_preds, _ = self.model.module.encode_image(input, return_dense=True)
            else:
                image_preds = self.model.module.encode_image(input)
            image_preds = image_preds / \
                (image_preds.norm(dim=-1, keepdim=True))
            logits = image_preds @ label_text_preds.t()
            scores = F.softmax(logits, dim=1) @ label_text_ensemble_matrix
            # compute prediction
            _, preds = logits.data.topk(k=1, dim=1)
            preds = preds.view(-1)
            # update batch information
            batch.update({'prediction': preds})
            batch.update({'score': scores})
            # save prediction information
            val_data['loader'].dataset.dump(writer, batch)

        writer.close()
        link.barrier()
        if self.dist.rank == 0:
            metrics = val_data['loader'].dataset.evaluate(res_file)
            self.logger.critical(json.dumps(metrics.metric, indent=2))
        else:
            metrics = {}
        link.barrier()
        # broadcast metrics to other process
        metrics = broadcast_object(metrics)
        self.model.train()
        return metrics


@link_dist
def main():
    parser = argparse.ArgumentParser(description='Classification Solver')
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--evaluate', action='store_true')

    args = parser.parse_args()
    # build solver
    import prototype.solver.crash_on_ipy
    solver = ClsSolver(args.config)
    # evaluate or train
    if args.evaluate:
        if not hasattr(solver.config.saver, 'pretrain'):
            solver.logger.warn(
                'Evaluating without resuming any solver checkpoints.')
        for id, val_data in enumerate(solver.val_data):
            solver.evaluate(val_data)
            if solver.ema is not None:
                solver.ema.load_ema(solver.model)
                solver.evaluate(val_data)
    else:
        if solver.config.data.last_iter < solver.config.data.max_iter:
            solver.train()
        else:
            solver.logger.info('Training has been completed to max_iter!')


if __name__ == '__main__':
    main()
