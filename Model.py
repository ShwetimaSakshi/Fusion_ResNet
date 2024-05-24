### YOUR CODE HERE
import torch
import os, time
import numpy as np
from Network import MyNetwork
from ImageUtils import parse_record
from tqdm import tqdm
import tensorflow as tf

"""This script defines the training, validation and testing process.
"""

class MyModel(torch.nn.Module):

    def __init__(self, configs, chckpoint_path=None):
        super(MyModel, self).__init__()
        self.configs = configs
        self.network = MyNetwork(configs)
        if chckpoint_path is not None:
            if not os.path.exists(chckpoint_path):
                raise Exception("Checkpoint path does not exit: ", chckpoint_path)
            self.load(chckpoint_path)

    def model_setup(self,training_configs):
        self.loss = torch.nn.CrossEntropyLoss()
        #self.optimizer = torch.optim.Adam(
        #    self.network.parameters(),
        #    lr=training_configs["learning_rate"],
        #    weight_decay = training_configs["weight_decay"]
        #)
        
        #self.optimizer = torch.optim.RMSprop(
        #    self.network.parameters(),
        #    lr=training_configs["learning_rate"],
        #    weight_decay = training_configs["weight_decay"]
        #)
        
        #self.optimizer = torch.optim.SGD(
        #    self.network.parameters(),
        #    lr=training_configs["learning_rate"],
        #    weight_decay = training_configs["weight_decay"],
        #    momentum=training_configs["momentum"],
        #    nesterov=True  # Use Nesterov momentum
        #)
        
        self.optimizer = torch.optim.Adamax(
            self.network.parameters(),
            lr=training_configs["learning_rate"],
            weight_decay = training_configs["weight_decay"]
        )
            

    def train(self, x_train, y_train, training_configs, x_valid=None, y_valid=None):
        self.network.train()

        num_of_samples = x_train.shape[0]
        num_of_batches = num_of_samples //training_configs["batch_size"] 
        torch.cuda.empty_cache()
        
        print('----------TRAINING----------')

        for epoch in range(1, training_configs["epochs"] + 1):
            start_time = time.time()

            shuffle_index = np.random.permutation(num_of_samples)
            epoch_x_train = x_train[shuffle_index]
            epoch_y_train = y_train[shuffle_index]

            new_learning_rate = 1e-4
            if(epoch % 10 == 0):
                new_learning_rate /= 10
            
            epoch_loss = 0
            self.optimizer.param_groups[0]['lr'] =  new_learning_rate

            for i in range(num_of_batches):

                #x_batch = [(parse_record(epoch_x_train[each_img], training=True)) for each_img in range(i * training_configs["batch_size"] , (i + 1) * training_configs["batch_size"])]
                x_batch = []
                for each_img in range(i * training_configs["batch_size"], (i + 1) * training_configs["batch_size"]):
                    # parse each record and apply sample pairing
                    image = parse_record(epoch_x_train[each_img], training=True)

                    # sample pairing
                    #if np.random.rand() > 0.5:
                    #    index = np.random.randint(0, len(epoch_x_train))
                    #    paired_image = parse_record(epoch_x_train[index], training=True)
                    #    alpha = np.random.uniform(0.3, 0.7)
                    #    image = alpha * image + (1 - alpha) * paired_image
#
                    x_batch.append(image)
                y_batch = epoch_y_train[i * training_configs["batch_size"] : (i + 1) * training_configs["batch_size"]]

                self.optimizer.zero_grad()
                # print('x_batch:', torch.tensor(np.array(x_batch), dtype=torch.float))
                output = self.network(torch.tensor(np.array(x_batch), dtype=torch.float).to('cuda'))
                loss = self.loss(output, torch.tensor(np.array(y_batch), dtype=torch.long).to('cuda'))
                
                l2_regularization = 0.0
                for param in self.network.parameters():
                    l2_regularization = l2_regularization + param.norm(2)
                loss = loss + 0.0001 * l2_regularization               
                
                loss.backward()
                self.optimizer.step()
                print('Batch {:d}/{:d} Loss {:.6f}'.format(i, num_of_batches, loss), end='\r', flush=True)
                epoch_loss = epoch_loss + loss
            
            duration = time.time() - start_time
            print('Epoch {:d} Loss {:.6f} Duration {:.3f} seconds.'.format(epoch, epoch_loss/(num_of_batches if num_of_batches>= 1 else 1), duration))

            if epoch % 10 == 0:
                self.save(epoch)
        
    def evaluate(self, x, y,checkpoint_num_list):
        self.network.eval()
        print('----------TEST OR VALIDATION----------')
        for checkpoint_num in checkpoint_num_list:
            checkpointfile = os.path.join(self.configs['save_dir'], 'model-%d.ckpt' % (checkpoint_num))
            if not os.path.exists(checkpointfile):
                raise FileNotFoundError(f"Checkpoint file {checkpointfile} not found.")
            else:
                self.load(checkpointfile)

            predicted_labels = []

            for i in tqdm(range(x.shape[0])):
                with torch.no_grad():
                    input_data = parse_record(x[i], False).reshape(1,3,32,32)
                    input_data_tensor = torch.tensor(input_data,dtype=torch.float).to('cuda')
                    output_data = self.network(input_data_tensor)
                    predicted_values = torch.argmax(output_data, dim=1)
                    predicted_labels.append(predicted_values)

            y = torch.tensor(y)
            preds = torch.tensor(predicted_labels)
            print('Test accuracy: {:.4f}'.format(torch.sum(preds == y) / y.shape[0]))
   
    def predict_prob(self, x,checkpoint_num):
        self.network.eval()
        checkpointfile = os.path.join(self.configs['save_dir'], 'model-%d.ckpt' % (checkpoint_num))
        if not os.path.exists(checkpointfile):
            raise FileNotFoundError(f"Checkpoint file {checkpointfile} not found.")
        else:
            self.load(checkpointfile)
        predicted_labels =[]

        for i in tqdm(range(x.shape[0])):
            with torch.no_grad():
                input_data = parse_record(x[i], False).reshape(1,3,32,32)
                input_data_tensor = torch.tensor(input_data,dtype=torch.float).to('cuda')
                predicted_values = self.network(input_data_tensor).cpu().numpy()
                predicted_labels.extend(predicted_values)
        
        predicted_labels = np.asarray(predicted_labels)
        max_value = np.max(predicted_labels, axis=-1, keepdims=True)
        exp_values = np.exp(predicted_labels - max_value)
        sum_exp_values = np.sum(exp_values, axis=-1, keepdims=True)
        log_sum_exp_values = max_value + np.log(sum_exp_values)

        pred_probs = exp_values / sum_exp_values
        return pred_probs
    
    # To save checkpoints
    def save(self, epoch):
        checkpoint_path = os.path.join(self.configs['save_dir'], 'model-%d.ckpt' % (epoch))
        os.makedirs(self.configs['save_dir'], exist_ok=True)
        torch.save(self.network.state_dict(), checkpoint_path)
        print("Checkpoint has been created.")

    # To load checkpoints
    def load(self, checkpoint_name):
        ckpt = torch.load(checkpoint_name, map_location="cpu")
        self.network.load_state_dict(ckpt, strict=True)
        print("Restored model parameters from {}".format(checkpoint_name))


### END CODE HERE
