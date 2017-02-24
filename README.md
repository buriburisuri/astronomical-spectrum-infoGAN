# InfoGAN analysis of astronomical spectrum data 


## Dependencies ( VERSION MUST BE MATCHED EXACTLY! )

1. tensorflow == 1.0.0
1. [sugartensor](https://github.com/buriburisuri/sugartensor) == 1.0.0.0

## Training the network

Execute
<pre><code>
python train.py
</code></pre>
to train the network. You can see the result ckpt files and log files in the 'asset/train' directory.
Launch tensorboard --logdir asset/train/log to monitor training process.

## Generating sample spectrum data
 
Execute
<pre><code>
python generate.py
</code></pre>
to generate sample time series data.  The graph image of generated time series data will be saved in the 'asset/train' directory.


# Authors
Namju Kim (namju.kim@kakaobrain.com) at KakaoBrain Corp.
Spencer Bialek
