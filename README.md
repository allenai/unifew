# UniFew
UniFew: Unified Few-shot Learning Model


## Installation

The main dependeny of this code is the `fewshot` package in the enclosed `flex` repo.  
Please first follow [flex installation instructions](https://github.com/allenai/flex#installation) to install the `fewshot` pacakge.
Additional dependencies are in the `requirements`.

```bash
git clone git@github.com:allenai/unifew.git
cd unifew

# optionally create a virtualenv with conda
conda create --name unifew python=3.8
# activate
conda activate unifew

# install flex from the flex repo
mkdir dependencies && cd dependencies
git clone git@github.com:allenai/flex.git
cd flex && pip install -e .

# then install main requirements
cd ../..
pip install -r requirements.txt
```

## Meta-testing on Flex

You can meta-test the model on the `flex` benchmark with the following command:
```bash
CUDA_VISIBLE_DEVICES=0 python test.py challenge=flex +hydra.run_dir=output/
```

This will run the model and save predictions in the `output/` directory.

You can use multiple GPUs to predict on different slices of the fleet challenge by specifing additional `start` and `stop` arguments.

If you wish to use a model that you have previously meta-trained, simply provide the relevant `ckpt_path=/path/to/checkpoint.ckpt` argument.

### Speed up meta-testing by parallelizing predictions

Meta-testing on single GPU can be slow.
Depending on number of available GPUs you can manually devide the meta-test episodes between the gpu and get corresponding predictions for each split.  
To do so, you can use the `start` and `end` arguments to the `test.py` script.  
Let's say we have 4 gpus. We launch the following commands.  

```bash
CUDA_VISIBLE_DEVICES=0 python test.py challenge=flex +hydra.run_dir=output/ start=0 stop=500  
CUDA_VISIBLE_DEVICES=1 python test.py challenge=flex +hydra.run_dir=output/ start=500 stop=1000
CUDA_VISIBLE_DEVICES=2 python test.py challenge=flex +hydra.run_dir=output/ start=1000 stop=1500
CUDA_VISIBLE_DEVICES=3 python test.py challenge=flex +hydra.run_dir=output/ start=1500
```

This would run the specified meta-test episodes for each gpu.  
You would end up with four different prediction files in the output directory with name `predictions_[start]-[stop]`.  

You can merge these prodections with `fewshot merge` command from the `flex` repo:  
`fewshot merge predictions_* predictions-merged.json`  

Then you can score predictions with the flex package:  
https://github.com/allenai/flex#result-validation-and-scoring 


## Meta-traininig the model

To meta-train the model, use the following command:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py hydra.run.dir=tmp/model/ trainer.max_steps=30000 query_batch_size=4 +sampler.min_way=2
```

Additional arguments can be found under 'conf/train.yaml' and 'conf/test.yaml'.
