	��C�l�"@��C�l�"@!��C�l�"@	U�H`t��?U�H`t��?!U�H`t��?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$��C�l�"@bX9�ȶ?A����Mb"@YZd;�O��?*	      O@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�I+��?!���{�A@)�� �rh�?1k���Zk;@:Preprocessing2F
Iterator::Model9��v���?!��{��D@)���Q��?12�c�18@:Preprocessing2U
Iterator::Model::ParallelMapV2�I+��?!���{�1@)�I+��?1���{�1@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap;�O��n�?!	!�B-@)�~j�t�x?1[k���Z#@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip;�O��n�?!	!�BM@){�G�zt?1!�B! @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor{�G�zt?!!�B! @){�G�zt?1!�B! @:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice�~j�t�h?![k���Z@)�~j�t�h?1[k���Z@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.5% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9V�H`t��?#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	bX9�ȶ?bX9�ȶ?!bX9�ȶ?      ��!       "      ��!       *      ��!       2	����Mb"@����Mb"@!����Mb"@:      ��!       B      ��!       J	Zd;�O��?Zd;�O��?!Zd;�O��?R      ��!       Z	Zd;�O��?Zd;�O��?!Zd;�O��?JCPU_ONLYYV�H`t��?b 