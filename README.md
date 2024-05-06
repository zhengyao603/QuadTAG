# QuadTAG

This is the source code of CSIT6000R Natural Language Processing group project of Group7.

## 1. Training

```python
python main.py --do_train --output_dir ./checkpoints/QuadTAG 
```

You can refer to `main.py` to check the optional arguments.

You can get our trained model and predicted results from [here](https://drive.google.com/drive/folders/1YNY29XDqzyA6RYKTAOe0JGPrjgmgmXiH).

## 2. Evaluation

```python
python main.py --do_dev --do_test --output_dir ./checkpoints/QuadTAG 
```

You can refer to `main.py` to check the optional arguments.

You can just set `--do_dev` or `--do_test` to evaluate the model on the development dataset or the test dataset respectively.

After you get the predicted results, you can run the below command to calculate the `precision`, `recall` and `F1 score` of the trained model. (Note that we have uploaded the predicted results to the `results` directory.)

```python
python eval_quad.py
```

