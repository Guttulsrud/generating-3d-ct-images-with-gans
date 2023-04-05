
if __name__ == "__main__":
    import os
    from monai.apps.auto3dseg import AutoRunner

    work_dir = "./kek2"
    if not os.path.isdir(work_dir):
        os.makedirs(work_dir)

    runner = AutoRunner(
        algos="segresnet",
        ensemble=False,
        work_dir=work_dir,
        input='input.yaml',
    )

    train_param = {
        "CUDA_VISIBLE_DEVICES": [0],  # use only 1 gpu

        #     "num_epochs_per_validation": 1,
        "num_images_per_batch": 1,
        #     "num_epochs": 1,
        #     "num_warmup_epochs": 1,
    }
    runner.set_training_params(train_param)
    runner.set_num_fold(num_fold=1)

    runner.run()