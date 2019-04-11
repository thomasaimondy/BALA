

## mk env path

```sh
virtualenv env --no-site-packages

source env/bin/activate

deactivate
```

## Conda env
```sh
conda create -n XXname python=2.7
conda env export > environment.yml
conda create --name myclone --clone myenv
conda activate XXname
conda deactivate
conda list  
conda remove --name myenv --all
conda info --envs
```

## Tensorflow Tips:
```sh
sudo pip install  https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.8.0-cp27-none-linux_x86_64.whl

sudo pip install  https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.8.0-cp27-none-linux_x86_64.whl

pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.8.0-cp27-none-linux_x86_64.whl
```

## Protobuf Error Tips:
The version of opencv should be 3.1.0 which fits for the protobuf version 3.0.0 in tensorflow 0.8.0

```##sh
conda install opencv=3.1.0
```
## skimage Error Tips:
```sh
conda install scikit-image
```
## caffe-gpu
tensorflow ok: video2textclone
caffe-gpu must installed before tensorflow-gpu



## cudaSuccess (35 vs. 0)
reinstall cuda driveer
conda install cudatoolkit=9.0

## No module named models.rnn
update into new tensowflow:
error1:
tf.concat([padding,output1],1) replace tf.concat(1, [padding,output1])
eror2: 
state1 = (tf.zeros([1,self.dim_hidden]),)*2
replace: state1 = tf.zeros([1, self.lstm1.state_size])
