# background
toy problem training binary classifier on a unit square without dropout, then using drop out at inference time.

# install
conda create --name pytorch_env pytorch::pytorch nvidia::cudatoolkit conda-forge::numpy matplotlib

# run
conda activate pytorch_env
python main.py

# results
![](results/no_dropout.png)
![](results/dropout.png)
![](results/dropout_0.001.png)
![](results/dropout_0.005.png)
![](results/dropout_0.010.png)
![](results/dropout_0.050.png)
![](results/dropout_0.100.png)
![](results/dropout_0.500.png)
![](results/dropout_1.000.png)
