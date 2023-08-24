# Testbed for Ray Tune and PyTorch Lightning

## Excerpt Sample Run of `ray-tune-torch-lightning-regression.py`

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

## Effect of cpu resources on  `ray-tune-torch-lightning-classification.py` run-time

### cpus_per_trial=1
```text
Trial status: 8 RUNNING | 7 TERMINATED
Current time: 2023-08-24 03:12:32. Total running time: 30s
Logical resource usage: 8.0/12 CPUs, 0/0 GPUs
Current best trial: fea4d_00000 with loss=4.818446636199951 and params={'layer_1_size': 64, 'layer_2_size': 256, 'lr': 0.001396043036024669, 'batch_size': 64}
╭──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Trial name                          status         layer_1_size     layer_2_size            lr     batch_size     iter     total time (s)       loss │
├──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ train_regression_tune_fea4d_00000   RUNNING                  64              256   0.00139604              64        2           19.1718     4.81845 │
│ train_regression_tune_fea4d_00001   RUNNING                  64              256   0.000985754            128        2           18.832      5.17938 │
│ train_regression_tune_fea4d_00002   RUNNING                  32              128   0.00107128             128        2           18.5043     5.07797 │
│ train_regression_tune_fea4d_00007   RUNNING                  64               64   0.044344               128        2           18.1705     7.26886 │
│ train_regression_tune_fea4d_00008   RUNNING                  64               64   0.000584723             32        2           18.2276     7.11105 │
│ train_regression_tune_fea4d_00012   RUNNING                 128              128   0.00818234             128        1            7.97265    7.92793 │
│ train_regression_tune_fea4d_00013   RUNNING                  32               64   0.0161633              128        1            7.49219    7.32777 │
│ train_regression_tune_fea4d_00014   RUNNING                  32              128   0.00349458              64        1            7.53948    6.09504 │
│ train_regression_tune_fea4d_00003   TERMINATED               32              128   0.000734132             64        1           10.7247    10.4131  │
│ train_regression_tune_fea4d_00004   TERMINATED              128              128   0.00838204              64        2           19.2182     6.65168 │
│ train_regression_tune_fea4d_00005   TERMINATED              128              256   0.000343951             32        2           19.5311     7.06071 │
│ train_regression_tune_fea4d_00006   TERMINATED               64              256   0.000263653             32        1           11.6308    10.52    │
│ train_regression_tune_fea4d_00009   TERMINATED               32              256   0.000112059             32        1           11.1778    99.6956  │
│ train_regression_tune_fea4d_00010   TERMINATED               32               64   0.0831065               64        2           18.3953     8.3339  │
│ train_regression_tune_fea4d_00011   TERMINATED               32               64   0.0799819               32        1           10.788     12.3104  │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

Trial status: 15 TERMINATED
Current time: 2023-08-24 03:13:01. Total running time: 1min 0s
Logical resource usage: 1.0/12 CPUs, 0/0 GPUs
Current best trial: fea4d_00002 with loss=2.3926172256469727 and params={'layer_1_size': 32, 'layer_2_size': 128, 'lr': 0.0010712805208237092, 'batch_size': 128}
╭──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Trial name                          status         layer_1_size     layer_2_size            lr     batch_size     iter     total time (s)       loss │
├──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ train_regression_tune_fea4d_00000   TERMINATED               64              256   0.00139604              64       10            52.1427    2.81955 │
│ train_regression_tune_fea4d_00001   TERMINATED               64              256   0.000985754            128        8            45.0255    3.09841 │
│ train_regression_tune_fea4d_00002   TERMINATED               32              128   0.00107128             128       10            50.8188    2.39262 │
│ train_regression_tune_fea4d_00003   TERMINATED               32              128   0.000734132             64        1            10.7247   10.4131  │
│ train_regression_tune_fea4d_00004   TERMINATED              128              128   0.00838204              64        2            19.2182    6.65168 │
│ train_regression_tune_fea4d_00005   TERMINATED              128              256   0.000343951             32        2            19.5311    7.06071 │
│ train_regression_tune_fea4d_00006   TERMINATED               64              256   0.000263653             32        1            11.6308   10.52    │
│ train_regression_tune_fea4d_00007   TERMINATED               64               64   0.044344               128        4            29.3428    6.70298 │
│ train_regression_tune_fea4d_00008   TERMINATED               64               64   0.000584723             32       10            50.419     2.94818 │
│ train_regression_tune_fea4d_00009   TERMINATED               32              256   0.000112059             32        1            11.1778   99.6956  │
│ train_regression_tune_fea4d_00010   TERMINATED               32               64   0.0831065               64        2            18.3953    8.3339  │
│ train_regression_tune_fea4d_00011   TERMINATED               32               64   0.0799819               32        1            10.788    12.3104  │
│ train_regression_tune_fea4d_00012   TERMINATED              128              128   0.00818234             128        2            14.1594    7.00171 │
│ train_regression_tune_fea4d_00013   TERMINATED               32               64   0.0161633              128        4            22.7276    5.6171  │
│ train_regression_tune_fea4d_00014   TERMINATED               32              128   0.00349458              64        4            22.8713    4.53997 │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

resources for each trial: {'cpu': 1, 'gpu': 0}
Best hyperparameters found were:
  loss metric: 2.3926172256469727
  config: {'layer_1_size': 32, 'layer_2_size': 128, 'lr': 0.0010712805208237092, 'batch_size': 128}
exiting train_regression_asha after 63.816086769104004 seconds
Done!
```


### cpu_per_trial=2
```text
Trial status: 6 RUNNING | 9 TERMINATED
Current time: 2023-08-24 03:16:55. Total running time: 30s
Logical resource usage: 12.0/12 CPUs, 0/0 GPUs
Current best trial: 9b9f2_00000 with loss=4.613883972167969 and params={'layer_1_size': 64, 'layer_2_size': 128, 'lr': 0.0005423496966102566, 'batch_size': 128}
╭──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Trial name                          status         layer_1_size     layer_2_size            lr     batch_size     iter     total time (s)       loss │
├──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ train_regression_tune_9b9f2_00000   RUNNING                  64              128   0.00054235             128        5           24.7668     4.61388 │
│ train_regression_tune_9b9f2_00007   RUNNING                 128              256   0.00243794              32        3           15.1779     4.85809 │
│ train_regression_tune_9b9f2_00010   RUNNING                 128              256   0.000243538             64        1            5.09257    8.72871 │
│ train_regression_tune_9b9f2_00012   RUNNING                  32               64   0.039086               128        1            4.53712    8.16584 │
│ train_regression_tune_9b9f2_00013   RUNNING                  64              256   0.00013428              32                                        │
│ train_regression_tune_9b9f2_00014   RUNNING                  64              256   0.0078452               32                                        │
│ train_regression_tune_9b9f2_00001   TERMINATED              128              256   0.0178175              128        1            6.94498   10.0199  │
│ train_regression_tune_9b9f2_00002   TERMINATED               32              128   0.0812579               64        1            6.26821   10.4934  │
│ train_regression_tune_9b9f2_00003   TERMINATED               64              256   0.00582813             128        2           11.2851     7.76556 │
│ train_regression_tune_9b9f2_00004   TERMINATED               64              128   0.046407               128        1            6.28664   11.5679  │
│ train_regression_tune_9b9f2_00005   TERMINATED              128              256   0.000410471             32        4           21.6249     5.38818 │
│ train_regression_tune_9b9f2_00006   TERMINATED              128               64   0.000332906             32        2            9.54079    7.62721 │
│ train_regression_tune_9b9f2_00008   TERMINATED               64              256   0.0100223               64        2            9.57253    8.71175 │
│ train_regression_tune_9b9f2_00009   TERMINATED              128              256   0.0271984               64        1            5.316     10.3099  │
│ train_regression_tune_9b9f2_00011   TERMINATED              128              128   0.0270899              128        1            4.93732    9.17452 │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯


Trial status: 15 TERMINATED
Current time: 2023-08-24 03:17:11. Total running time: 46s
Logical resource usage: 2.0/12 CPUs, 0/0 GPUs
Current best trial: 9b9f2_00000 with loss=3.102931022644043 and params={'layer_1_size': 64, 'layer_2_size': 128, 'lr': 0.0005423496966102566, 'batch_size': 128}
╭──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Trial name                          status         layer_1_size     layer_2_size            lr     batch_size     iter     total time (s)       loss │
├──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ train_regression_tune_9b9f2_00000   TERMINATED               64              128   0.00054235             128       10           41.2741     3.10293 │
│ train_regression_tune_9b9f2_00001   TERMINATED              128              256   0.0178175              128        1            6.94498   10.0199  │
│ train_regression_tune_9b9f2_00002   TERMINATED               32              128   0.0812579               64        1            6.26821   10.4934  │
│ train_regression_tune_9b9f2_00003   TERMINATED               64              256   0.00582813             128        2           11.2851     7.76556 │
│ train_regression_tune_9b9f2_00004   TERMINATED               64              128   0.046407               128        1            6.28664   11.5679  │
│ train_regression_tune_9b9f2_00005   TERMINATED              128              256   0.000410471             32        4           21.6249     5.38818 │
│ train_regression_tune_9b9f2_00006   TERMINATED              128               64   0.000332906             32        2            9.54079    7.62721 │
│ train_regression_tune_9b9f2_00007   TERMINATED              128              256   0.00243794              32        8           34.2789     3.91423 │
│ train_regression_tune_9b9f2_00008   TERMINATED               64              256   0.0100223               64        2            9.57253    8.71175 │
│ train_regression_tune_9b9f2_00009   TERMINATED              128              256   0.0271984               64        1            5.316     10.3099  │
│ train_regression_tune_9b9f2_00010   TERMINATED              128              256   0.000243538             64        2           10.1425     7.58777 │
│ train_regression_tune_9b9f2_00011   TERMINATED              128              128   0.0270899              128        1            4.93732    9.17452 │
│ train_regression_tune_9b9f2_00012   TERMINATED               32               64   0.039086               128        4           15.8441     6.2851  │
│ train_regression_tune_9b9f2_00013   TERMINATED               64              256   0.00013428              32        1            4.78974   17.309   │
│ train_regression_tune_9b9f2_00014   TERMINATED               64              256   0.0078452               32        4           15.0498     6.44857 │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

resources for each trial: {'cpu': 2, 'gpu': 0}
Best hyperparameters found were:
  loss metric: 3.102931022644043
  config: {'layer_1_size': 64, 'layer_2_size': 128, 'lr': 0.0005423496966102566, 'batch_size': 128}
exiting train_regression_asha after 50.878902435302734 seconds
Done!

```

### cpu_per_trial=4

```text
Trial status: 8 TERMINATED | 6 RUNNING | 1 PENDING
Current time: 2023-08-24 03:20:58. Total running time: 30s
Logical resource usage: 12.0/12 CPUs, 0/0 GPUs
Current best trial: 2c67d_00008 with loss=4.516640663146973 and params={'layer_1_size': 128, 'layer_2_size': 128, 'lr': 0.0013577274843424307, 'batch_size': 64}
╭──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Trial name                          status         layer_1_size     layer_2_size            lr     batch_size     iter     total time (s)       loss │
├──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ train_regression_tune_2c67d_00004   RUNNING                  64              128   0.00977846             128        5           24.7063     5.4254  │
│ train_regression_tune_2c67d_00007   RUNNING                 128              256   0.00501308              64        3           15.4439     6.4488  │
│ train_regression_tune_2c67d_00008   RUNNING                 128              128   0.00135773              64        3           14.4063     4.51664 │
│ train_regression_tune_2c67d_00010   RUNNING                  64              256   0.000710557            128        3           14.1417     5.10094 │
│ train_regression_tune_2c67d_00012   RUNNING                  32               64   0.00411967              32        3           13.4673     4.57741 │
│ train_regression_tune_2c67d_00013   RUNNING                 128              256   0.000525587            128        1            5.04017    8.22284 │
│ train_regression_tune_2c67d_00000   TERMINATED              128              128   0.000248698             32        1            6.42915    8.94751 │
│ train_regression_tune_2c67d_00001   TERMINATED               64              256   0.00966153              32        2           11.2803     6.99245 │
│ train_regression_tune_2c67d_00002   TERMINATED               64              128   0.0337406               64        1            6.18338   12.0615  │
│ train_regression_tune_2c67d_00003   TERMINATED              128              256   0.000116656             32        1            6.70216   12.3318  │
│ train_regression_tune_2c67d_00005   TERMINATED               64              256   0.0835432              128        1            6.57966    8.90171 │
│ train_regression_tune_2c67d_00006   TERMINATED               32              256   0.00061776              32        1            4.65402   10.6033  │
│ train_regression_tune_2c67d_00009   TERMINATED               64              256   0.0570869              128        1            4.94391   13.0592  │
│ train_regression_tune_2c67d_00011   TERMINATED               64              256   0.0123053              128        1            4.87301    9.79954 │
│ train_regression_tune_2c67d_00014   PENDING                  32               64   0.000725013            128                                        │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

Trial status: 15 TERMINATED
Current time: 2023-08-24 03:21:22. Total running time: 54s
Logical resource usage: 2.0/12 CPUs, 0/0 GPUs
Current best trial: 2c67d_00010 with loss=2.812286615371704 and params={'layer_1_size': 64, 'layer_2_size': 256, 'lr': 0.000710557104067881, 'batch_size': 128}
╭──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Trial name                          status         layer_1_size     layer_2_size            lr     batch_size     iter     total time (s)       loss │
├──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ train_regression_tune_2c67d_00000   TERMINATED              128              128   0.000248698             32        1            6.42915    8.94751 │
│ train_regression_tune_2c67d_00001   TERMINATED               64              256   0.00966153              32        2           11.2803     6.99245 │
│ train_regression_tune_2c67d_00002   TERMINATED               64              128   0.0337406               64        1            6.18338   12.0615  │
│ train_regression_tune_2c67d_00003   TERMINATED              128              256   0.000116656             32        1            6.70216   12.3318  │
│ train_regression_tune_2c67d_00004   TERMINATED               64              128   0.00977846             128       10           42.5862     4.70379 │
│ train_regression_tune_2c67d_00005   TERMINATED               64              256   0.0835432              128        1            6.57966    8.90171 │
│ train_regression_tune_2c67d_00006   TERMINATED               32              256   0.00061776              32        1            4.65402   10.6033  │
│ train_regression_tune_2c67d_00007   TERMINATED              128              256   0.00501308              64        4           20.5975     5.89365 │
│ train_regression_tune_2c67d_00008   TERMINATED              128              128   0.00135773              64       10           40.5308     3.35407 │
│ train_regression_tune_2c67d_00009   TERMINATED               64              256   0.0570869              128        1            4.94391   13.0592  │
│ train_regression_tune_2c67d_00010   TERMINATED               64              256   0.000710557            128       10           38.46       2.81229 │
│ train_regression_tune_2c67d_00011   TERMINATED               64              256   0.0123053              128        1            4.87301    9.79954 │
│ train_regression_tune_2c67d_00012   TERMINATED               32               64   0.00411967              32       10           36.9054     3.45425 │
│ train_regression_tune_2c67d_00013   TERMINATED              128              256   0.000525587            128        2           10.0112     6.596   │
│ train_regression_tune_2c67d_00014   TERMINATED               32               64   0.000725013            128        1            3.7865    11.2807  │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

resources for each trial: {'cpu': 2, 'gpu': 0}
Best hyperparameters found were:
  loss metric: 2.812286615371704
  config: {'layer_1_size': 64, 'layer_2_size': 256, 'lr': 0.000710557104067881, 'batch_size': 128}
exiting train_regression_asha after 58.7718563079834 seconds
Done!
```