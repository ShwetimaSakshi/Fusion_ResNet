# Below configures are examples, 
# you can modify them as you wish.

### YOUR CODE HERE

model_configs = {
	"name": 'MyModel',
	"save_dir": '../saved_models/',
	"depth": 28, 
    "result_dir": '../../',
    "data_dir": '../',
    "num_classes": 10,
    "cardinality": 10,
    "dropRate": 0.2
	# ...
}

training_configs = {
	"learning_rate": 0.01,
    "batch_size": 64,
	"epochs": 60,
	"weight_decay":5e-4,
	"momentum":0.9
	# ...
 
}

### END CODE HERE
