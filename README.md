# unifew
Unifew: Unified Fewshot Learning Model


## Dependencies

The main dependeny of this code is the `fewshot` package in the enclosed `fleet` repo. 
Please follow instructions in `fleet` to first install the `fewshot` package.
Additional dependencies are in the `requirements`

```bash
# First install `fewshot` package from `fleet`
# then install additional requirements
pip install -r requirements.txt
```

## Run the model on meta-test mode

You can run the model on `fleet` benchmark with the following command:
```bash
CUDA_VISIBLE_DEVICES=0 python test.py challenge=fleet +hydra.run_dir=output/
```

This will run the model and save predictions in the `output/` directory.

You can use multiple GPUs to predict on different slices of the fleet challenge by specifing additional `start` and `stop` arguments.

If you wish to use a model that you have previously meta-trained, simply provide the relevant `ckpt_path=/path/to/checkpoint.ckpt` argument.

## Meta-train the model


```bash
CUDA_VISIBLE_DEVICES=0 python train.py hydra.run.dir=tmp/model/ trainer.max_steps=30000 query_batch_size=4 +sampler.min_way=2
```

Additional arguments can be found under 'conf/train.yaml' and 'conf/test.yaml'.
