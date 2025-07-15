'''
by lyuwenyu
'''
import time
import json
import datetime
import random

import torch

from src.misc import dist
from src.data import get_coco_api_from_dataset

from .solver import BaseSolver
from .det_engine import train_one_epoch, evaluate
from src.transforms.transform import HistogramEqualization, LinearStretch, CLAHE, GPUHistogramEqualization, GPUCLAHE


class DetSolver(BaseSolver):
    
    def fit(self, ):
        print("Start training")
        # initialize model, dataloaders, optimizer, etc.
        self.train()

        args = self.cfg 
        
        # build enhanment
        use_gpu = torch.cuda.is_available()
        if args.enhance_method == 'he':
            if use_gpu:
                enhancer = GPUHistogramEqualization()
            else:
                enhancer = HistogramEqualization()
        elif args.enhance_method == 'linear':
            enhancer = LinearStretch()
        elif args.enhance_method == 'clahe':
            if use_gpu:
                enhancer = GPUCLAHE(args.clahe_clip, args.clahe_grid)
            else:
                enhancer = CLAHE(args.clahe_clip, args.clahe_grid)
        else:
            enhancer = None
            
        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        # best_stat = {'coco_eval_bbox': 0, 'coco_eval_masks': 0, 'epoch': -1, }
        best_stat = {'epoch': -1, }

        start_time = time.time()
        for epoch in range(self.last_epoch + 1, args.epoches):
            if dist.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            # decide per-epoch enhancement
            apply_enh = enhancer is not None and (random.random() <= args.he_threshold)
            print(f"[Epoch {epoch}] method={args.enhance_method}  applied={apply_enh}")

            train_stats = train_one_epoch(
                self.model,
                self.criterion,
                self.train_dataloader,
                self.optimizer,
                self.device,
                epoch,
                args.clip_max_norm,
                print_freq=args.log_step,
                ema=self.ema,
                scaler=self.scaler,
                apply_enh=apply_enh,
                enhancer=enhancer
            )

            self.lr_scheduler.step()
            
            if self.output_dir:
                checkpoint_paths = [self.output_dir / 'checkpoint.pth']
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.checkpoint_step == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    dist.save_on_master(self.state_dict(epoch), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            stat_val, test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                base_ds,
                self.device,
                self.output_dir,
                apply_enh=apply_enh,
                enhancer=enhancer
            )

            # TODO 
            for k in test_stats.keys():
                if k in best_stat:
                    best_stat['epoch'] = epoch if test_stats[k][0] > best_stat[k] else best_stat['epoch']
                    best_stat[k] = max(best_stat[k], test_stats[k][0])
                else:
                    best_stat['epoch'] = epoch
                    best_stat[k] = test_stats[k][0]
            print('best_stat: ', best_stat)


            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'val_{k}' : v for k,v in stat_val.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

            # add enhancement info to log
            log_stats['enhance_method'] = args.enhance_method
            log_stats['enhance_applied'] = apply_enh

            if self.output_dir and dist.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                    self.output_dir / "eval" / name)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))


    def val(self, ):
        self.eval()
        args = self.cfg 
        
        # build enhancement
        if args.enhance_method == 'he':
            enhancer = HistogramEqualization()
        elif args.enhance_method == 'linear':
            enhancer = LinearStretch()
        elif args.enhance_method == 'clahe':
            enhancer = CLAHE(args.clahe_clip, args.clahe_grid)
        else:
            enhancer = None

        apply_enh = enhancer is not None and (random.random() <= args.he_threshold)
        print(f"[Evaluate the test data set] method={args.enhance_method}  applied={apply_enh}")

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        
        module = self.ema.module if self.ema else self.model
        stat_val,test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, base_ds, self.device, self.output_dir, apply_enh=apply_enh,
                enhancer=enhancer)
                
        if self.output_dir:
            dist.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir /f"{self.output_dir.split('_')[4:]}_eval.pth")
        
        return
