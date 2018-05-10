
该目录包含用于创建 RNN 和 Seq2Seq 模型的功能。 
有关如何开始使用它们的详细说明，请参阅
[tutorials on tensorflow.org](http://tensorflow.org/tutorials/).

以下是对目录中内容的简要：

File         | What's in it?
------------ | -------------
`ptb/`       | PTB language model, see the [RNN Tutorial](http://tensorflow.org/tutorials/recurrent/)
`quickdraw/` | Quick, Draw! model, see the [RNN Tutorial for Drawing Classification](https://www.tensorflow.org/versions/master/tutorials/recurrent_quickdraw)

ptb_word_lm.py 运行前需要先下载 [PTB数据集](http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz)
运行命令：
```
cd models/tutorials/rnn/ptb
python ptb_word_lm.py --data_path=$HOME/simple-examples/data/ --model=small
```

If you're looking for the 
[`seq2seq` tutorial code](http://tensorflow.org/tutorials/seq2seq/), it lives
in [its own repo](https://github.com/tensorflow/nmt).