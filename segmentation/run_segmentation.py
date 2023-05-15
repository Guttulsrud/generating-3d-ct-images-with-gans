import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
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
        "CUDA_VISIBLE_DEVICES": [0],
        "num_images_per_batch": 1,

    }
    runner.set_training_params(train_param)
    runner.run()
