# Masked_llama

## how to use

run `pip install -r requirements.txt` to install all packages

chooose a model, then run 
`
torchrun --nproc_per_node 1 test.py     --ckpt_dir {$model_path}     --tokenizer_path {$tokenizer_path}     --max_seq_len 512 --max_batch_size 6 > output1.txt 2>&1
`
the output will be stored in output1.txt

### modified files

The modification is mainly in `./llama/model.py`---`Tranformer`---`forward` and `./llama/generate.py`---`generate`