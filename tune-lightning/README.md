# Testbed for Ray Tune and PyTorch Lightning

## Excerpt Sample Run of `ray-tune-torch-lightning-regression.py`

```text
```text
2023-08-23 11:25:39,023 WARNING services.py:1832 -- WARNING: The object store is using /tmp instead of /dev/shm because /dev/shm has only 66670592 bytes available. This will harm performance! You may be able to free up space by deleting files in /dev/shm. If you are inside a Docker container, you can increase /dev/shm size by passing '--shm-size=3.18gb' to 'docker run' (or add it to the run_options list in a Ray cluster config). Make sure to set this to more than 30% of available RAM.
2023-08-23 11:25:40,089 INFO worker.py:1621 -- Started a local Ray instance.
2023-08-23 11:25:41,355 INFO tune.py:226 -- Initializing Ray automatically. For cluster usage or custom Ray initialization, call `ray.init(...)` before `Tuner(...)`.
2023-08-23 11:25:41,357 INFO tune.py:666 -- [output] This will use the new output engine with verbosity 1. To disable the new output and use the legacy output engine, set the environment variable RAY_AIR_NEW_OUTPUT=0. For more information, please see https://github.com/ray-project/ray/issues/36949
2023-08-23 11:25:41,378 WARNING tune.py:997 -- AIR_VERBOSITY is set, ignoring passed-in ProgressReporter for now.
╭──────────────────────────────────────────────────────────╮
│ Configuration for experiment     tune_regression_asha    │
├──────────────────────────────────────────────────────────┤
│ Search algorithm                 BasicVariantGenerator   │
│ Scheduler                        AsyncHyperBandScheduler │
│ Number of trials                 15                      │
╰──────────────────────────────────────────────────────────╯

View detailed results here: /root/ray_results/tune_regression_asha
To visualize your results with TensorBoard, run: `tensorboard --logdir /root/ray_results/tune_regression_asha`

Trial status: 15 PENDING
Current time: 2023-08-23 11:25:41. Total running time: 0s
Logical resource usage: 10.0/12 CPUs, 0/0 GPUs
╭─────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Trial name                          status       layer_1_size     layer_2_size            lr     batch_size │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ train_regression_tune_cadb3_00000   PENDING                64              256   0.0966731              128 │
│ train_regression_tune_cadb3_00001   PENDING                32               64   0.0030709              128 │
│ train_regression_tune_cadb3_00002   PENDING                32              128   0.000712432            128 │
│ train_regression_tune_cadb3_00003   PENDING                32              256   0.00897816              32 │
│ train_regression_tune_cadb3_00004   PENDING               128               64   0.0763211               64 │
│ train_regression_tune_cadb3_00005   PENDING                64               64   0.000431196             64 │
│ train_regression_tune_cadb3_00006   PENDING                64              256   0.00653935             128 │
│ train_regression_tune_cadb3_00007   PENDING               128               64   0.00100462              32 │
│ train_regression_tune_cadb3_00008   PENDING                64               64   0.0120265              128 │
│ train_regression_tune_cadb3_00009   PENDING                64              256   0.00117377             128 │
│ train_regression_tune_cadb3_00010   PENDING                64              256   0.000205281             32 │
│ train_regression_tune_cadb3_00011   PENDING                32               64   0.000100197             32 │
│ train_regression_tune_cadb3_00012   PENDING               128               64   0.000345908             64 │
│ train_regression_tune_cadb3_00013   PENDING                32              128   0.000231343             32 │
│ train_regression_tune_cadb3_00014   PENDING                64               64   0.0098322               64 │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

Trial train_regression_tune_cadb3_00009 started with configuration:
╭────────────────────────────────────────────────────────────╮
│ Trial train_regression_tune_cadb3_00009 config             │
├────────────────────────────────────────────────────────────┤
│ batch_size                                             128 │
│ layer_1_size                                            64 │
│ layer_2_size                                           256 │
│ lr                                                 0.00117 │
╰────────────────────────────────────────────────────────────╯

<<<DELETED LINES>>>

Trial status: 14 TERMINATED | 1 RUNNING
Current time: 2023-08-23 11:26:11. Total running time: 30s
Logical resource usage: 1.0/12 CPUs, 0/0 GPUs
Current best trial: cadb3_00001 with loss=3.2582108974456787 and params={'layer_1_size': 32, 'layer_2_size': 64, 'lr': 0.0030709044950215166, 'batch_size': 128}
╭───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Trial name                          status         layer_1_size     layer_2_size            lr     batch_size     iter     total time (s)        loss │
├───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ train_regression_tune_cadb3_00001   RUNNING                  32               64   0.0030709              128        6           22.1109      3.25821 │
│ train_regression_tune_cadb3_00000   TERMINATED               64              256   0.0966731              128        1            8.28052    15.0024  │
│ train_regression_tune_cadb3_00002   TERMINATED               32              128   0.000712432            128        1            7.75519    10.473   │
│ train_regression_tune_cadb3_00003   TERMINATED               32              256   0.00897816              32        2           12.4848      6.79994 │
│ train_regression_tune_cadb3_00004   TERMINATED              128               64   0.0763211               64        1            8.21568    14.3156  │
│ train_regression_tune_cadb3_00005   TERMINATED               64               64   0.000431196             64        1            7.63368    10.8616  │
│ train_regression_tune_cadb3_00006   TERMINATED               64              256   0.00653935             128        1            7.97538     8.02104 │
│ train_regression_tune_cadb3_00007   TERMINATED              128               64   0.00100462              32        4           18.1214      4.1151  │
│ train_regression_tune_cadb3_00008   TERMINATED               64               64   0.0120265              128        2           12.6317      6.2848  │
│ train_regression_tune_cadb3_00009   TERMINATED               64              256   0.00117377             128        2           12.6097      5.30926 │
│ train_regression_tune_cadb3_00010   TERMINATED               64              256   0.000205281             32        1            7.91921    10.8161  │
│ train_regression_tune_cadb3_00011   TERMINATED               32               64   0.000100197             32        1            7.65976   147.559   │
│ train_regression_tune_cadb3_00012   TERMINATED              128               64   0.000345908             64        1            4.94285     9.27375 │
│ train_regression_tune_cadb3_00013   TERMINATED               32              128   0.000231343             32        1            4.75394    14.5319  │
│ train_regression_tune_cadb3_00014   TERMINATED               64               64   0.0098322               64        2            7.43764     5.99576 │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

Trial train_regression_tune_cadb3_00001 completed after 10 iterations at 2023-08-23 11:26:19. Total running time: 38s
╭────────────────────────────────────────────────────────────╮
│ Trial train_regression_tune_cadb3_00001 result             │
├────────────────────────────────────────────────────────────┤
│ time_this_iter_s                                   2.30295 │
│ time_total_s                                       31.3613 │
│ training_iteration                                      10 │
│ loss                                               2.84826 │
╰────────────────────────────────────────────────────────────╯

Trial status: 15 TERMINATED
Current time: 2023-08-23 11:26:19. Total running time: 38s
Logical resource usage: 1.0/12 CPUs, 0/0 GPUs
Current best trial: cadb3_00001 with loss=2.8482649326324463 and params={'layer_1_size': 32, 'layer_2_size': 64, 'lr': 0.0030709044950215166, 'batch_size': 128}
╭───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Trial name                          status         layer_1_size     layer_2_size            lr     batch_size     iter     total time (s)        loss │
├───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ train_regression_tune_cadb3_00000   TERMINATED               64              256   0.0966731              128        1            8.28052    15.0024  │
│ train_regression_tune_cadb3_00001   TERMINATED               32               64   0.0030709              128       10           31.3613      2.84826 │
│ train_regression_tune_cadb3_00002   TERMINATED               32              128   0.000712432            128        1            7.75519    10.473   │
│ train_regression_tune_cadb3_00003   TERMINATED               32              256   0.00897816              32        2           12.4848      6.79994 │
│ train_regression_tune_cadb3_00004   TERMINATED              128               64   0.0763211               64        1            8.21568    14.3156  │
│ train_regression_tune_cadb3_00005   TERMINATED               64               64   0.000431196             64        1            7.63368    10.8616  │
│ train_regression_tune_cadb3_00006   TERMINATED               64              256   0.00653935             128        1            7.97538     8.02104 │
│ train_regression_tune_cadb3_00007   TERMINATED              128               64   0.00100462              32        4           18.1214      4.1151  │
│ train_regression_tune_cadb3_00008   TERMINATED               64               64   0.0120265              128        2           12.6317      6.2848  │
│ train_regression_tune_cadb3_00009   TERMINATED               64              256   0.00117377             128        2           12.6097      5.30926 │
│ train_regression_tune_cadb3_00010   TERMINATED               64              256   0.000205281             32        1            7.91921    10.8161  │
│ train_regression_tune_cadb3_00011   TERMINATED               32               64   0.000100197             32        1            7.65976   147.559   │
│ train_regression_tune_cadb3_00012   TERMINATED              128               64   0.000345908             64        1            4.94285     9.27375 │
│ train_regression_tune_cadb3_00013   TERMINATED               32              128   0.000231343             32        1            4.75394    14.5319  │
│ train_regression_tune_cadb3_00014   TERMINATED               64               64   0.0098322               64        2            7.43764     5.99576 │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

Best hyperparameters found were:
  loss metric: 2.8482649326324463
  config: {'layer_1_size': 32, 'layer_2_size': 64, 'lr': 0.0030709044950215166, 'batch_size': 128}
Done!
(train_regression_tune pid=17930) >>>>17930 entering train_regression_tune with config: {'layer_1_size': 64, 'layer_2_size': 64, 'lr': 0.009832201966084881, 'batch_size': 64} [repeated 2x across cluster]
(train_regression_tune pid=17930) GPU available: False, used: False [repeated 2x across cluster]
(train_regression_tune pid=17930) TPU available: False, using: 0 TPU cores [repeated 2x across cluster]
(train_regression_tune pid=17930) IPU available: False, using: 0 IPUs [repeated 2x across cluster]
(train_regression_tune pid=17930) HPU available: False, using: 0 HPUs [repeated 2x across cluster]
(train_regression_tune pid=17930)  [repeated 2x across cluster]
(train_regression_tune pid=17930)   | Name    | Type   | Params [repeated 2x across cluster]
(train_regression_tune pid=17930) ----------------------------------- [repeated 4x across cluster]
(train_regression_tune pid=17930) 1 | layer_2 | Linear | 4.2 K  [repeated 4x across cluster]
(train_regression_tune pid=17930) 2 | layer_3 | Linear | 65     [repeated 2x across cluster]
(train_regression_tune pid=17930) 10.7 K    Trainable params [repeated 2x across cluster]
(train_regression_tune pid=17930) 0         Non-trainable params [repeated 2x across cluster]
(train_regression_tune pid=17930) 10.7 K    Total params [repeated 2x across cluster]
(train_regression_tune pid=17930) 0.043     Total estimated model params size (MB) [repeated 2x across cluster]
(train_regression_tune pid=17930) /opt/conda/lib/python3.10/site-packages/pytorch_lightning/trainer/connectors/data_connector.py:438: PossibleUserWarning: The dataloader, val_dataloader, does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` (try 12 which is the number of cpus on this machine) in the `DataLoader` init to improve performance. [repeated 2x across cluster]
(train_regression_tune pid=17930)   rank_zero_warn( [repeated 4x across cluster]
(train_regression_tune pid=17930) /opt/conda/lib/python3.10/site-packages/pytorch_lightning/trainer/connectors/data_connector.py:438: PossibleUserWarning: The dataloader, train_dataloader, does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` (try 12 which is the number of cpus on this machine) in the `DataLoader` init to improve performance. [repeated 2x
```