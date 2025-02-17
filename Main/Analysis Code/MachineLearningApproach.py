import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.python.ops.numpy_ops import np_config
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def find_folder(directory, subj_id):

    matching_folders = []
    for root, dirs, files in os.walk(directory):
        for folder_name in dirs:
            if subj_id in folder_name:
                matching_folders.append(os.path.join(root, folder_name))
    return matching_folders



np_config.enable_numpy_behavior()

File_Path = os.path.dirname(os.getcwd()) + '/Collected Data/Subjective Similarity Judgment Task'
Subj_ID = ['108169884-1','108169884-2',
           '695758433-1','695758433-2',
           '619502508-1','619502508-2',
           '584740030-1','584740030-2',
           '089858508-1','089858508-2',
           '801165888-1','801165888-2',
           '040062257-1','040062257-2',
           '940332894-1','940332894-2',
           '518402380-1','518402380-2',
           '415090147-1','415090147-2',
           '951506687-1','951506687-2',
           '682207766-1','682207766-2']


for Subj in range(len(Subj_ID)):

    file_path_subj = find_folder(File_Path, Subj_ID[Subj])

    #### read the data ####
    Click_Data = pd.read_csv(file_path_subj[0] + '\\' + 'click_data.csv')
    trial_stim = Click_Data.trial
    trial_stim = trial_stim.to_numpy()
    target_stim = Click_Data.target_stim
    target_stim = target_stim.to_numpy()
    clicked_stim = Click_Data.clicked_stim
    clicked_stim = clicked_stim.to_numpy()

    #### break down into sub-trials ####
    data = []
    for i in range(0, np.max(trial_stim)+1):
        indices = np.where(trial_stim == i)[0]
        if len(indices) == 4:
            Target = target_stim[indices[0]]
            for n in range(4):
                for m in range(n+1, 4):
                    CloseCandidate = clicked_stim[indices[n]]
                    FarCandidate = clicked_stim[indices[m]]
                    data.append([Target, CloseCandidate, FarCandidate, abs(n-m)])
    data = np.array(data)

    ###### define the ML model ######
    class MLModel(tf.keras.Model):
      def __init__(self, input_shape=(4,), l1_regularization=0.0001, l2_regularization=0.0001):
          super().__init__()
          self._input_shape = input_shape
          self.built = False
          self.l1_regularization = l1_regularization
          self.l2_regularization = l2_regularization

      def build(self):
          self.W = self.add_weight(shape=(30, 10),
                                  initializer=tf.keras.initializers.Constant(np.random.uniform(-1, 1, (30, 10))),
                                  trainable=True,
                                  regularizer=tf.keras.regularizers.L1L2(l1=self.l1_regularization, l2=self.l2_regularization)
                                  )
          self.built = True

      def call(self, inputs):
          target = tf.gather(self.W, tf.cast(inputs[:, 0], tf.int32))
          similar = tf.gather(self.W, tf.cast(inputs[:, 1], tf.int32))
          dissimilar = tf.gather(self.W, tf.cast(inputs[:, 2], tf.int32))
          dist_target_similar = tf.sqrt(tf.reduce_sum(tf.square(target - similar), axis=1))
          dist_target_dissimilar = tf.sqrt(tf.reduce_sum(tf.square(target - dissimilar), axis=1))
          p = tf.nn.sigmoid(inputs[:, 3] * (dist_target_similar - dist_target_dissimilar))
          return tf.expand_dims(p, axis=-1)

    L1_list = [0.00025]
    L2_list = [0.00025]
    #Accuracy = np.zeros((len(L1_list), len(L2_list)))
    #ValAccuracy = np.zeros((len(L1_list), len(L2_list)))
    for k1, L1 in enumerate(L1_list):
        for k2, L2 in enumerate(L2_list):

            output = np.zeros((data.shape[0], 1), dtype=int)
            X_train, X_val, y_train, y_val = train_test_split(data, output, test_size=0.2, random_state=42)
            ### have no validation ###
            X_train = np.concatenate((X_train, X_val), axis=0)
            y_train = np.concatenate((y_train, y_val), axis=0)


            ##### build and run the ML model #####
            input_shape = (4,)
            model = MLModel(input_shape, l1_regularization=L1, l2_regularization=L2)
            model.build()

            Optimizer = tf.keras.optimizers.Adam(learning_rate=0.01, clipvalue=0.5)
            Loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.1)

            Metrics = ['accuracy']
            model.compile(optimizer=Optimizer, loss=Loss, metrics=Metrics)

            history = model.fit(x=X_train,
                                y=y_train,
                                #validation_data=(X_val, y_val),
                                batch_size=1000,
                                epochs=1000)


            train_accuracy = history.history['accuracy']
            #val_accuracy = history.history['val_accuracy']

            ####### plot the accuarcy ########
            fig, ax = plt.subplots(1, 2, figsize=(14, 6))
            ax[0].plot(train_accuracy, label='Training Accuracy')
            #ax[0].plot(val_accuracy, label='Validation Accuracy', linestyle='--')
            ax[0].set_title('Accuracy per Epoch')
            ax[0].set_xlabel('Epochs')
            ax[0].set_ylabel('Accuracy')
            ax[0].legend()
            ax[0].grid(True)


            trained_embeddings = model.get_weights()[0]
            column_std = np.std(trained_embeddings, axis=0)
            ### sort according to the std ###
            sorted_indices = np.argsort(column_std)[::-1]
            trained_embeddings_sorted = trained_embeddings[:, sorted_indices]
            column_std = np.std(trained_embeddings_sorted, axis=0)

            #### evaluation of the dimensions #######
            ax[1].hist(column_std, bins=20, color='skyblue', edgecolor='black')
            ax[1].set_title('Histogram of Column Std')
            ax[1].set_xlabel('Standard Deviation')
            ax[1].set_ylabel('Frequency')
            ax[1].grid(True)

            folder_path_N = os.path.join(file_path_subj[0], 'ML_Approach')
            os.makedirs(folder_path_N, exist_ok=True)

            ##### save the plot #####
            plot_filename = os.path.join(folder_path_N, f'accuracy_and_histogram_{L1}_{L2}.png')
            plt.tight_layout()
            plt.savefig(plot_filename)
            plt.close()

            ##### save the embeddings #####
            save_filename = os.path.join(folder_path_N, f'Space_{L1}_{L2}.csv')
            df = pd.DataFrame(trained_embeddings_sorted)
            df.to_csv(save_filename, index=False)

            #Accuracy[k1, k2] = train_accuracy[-1]
            #ValAccuracy[k1, k2] = val_accuracy[-1]

    #plt.imshow(Accuracy, cmap='viridis', aspect='auto')
    #plt.colorbar()
    #plt.xticks(ticks=np.arange(len(L1_list)), labels=L1_list)
    #plt.yticks(ticks=np.arange(len(L2_list)), labels=L2_list)
    #plt.grid(color='black', linestyle='-', linewidth=1, alpha=0.5)
    #for i in range(Accuracy.shape[0]):  # Rows
    #    for j in range(Accuracy.shape[1]):  # Columns
    #        plt.text(j, i, f'{Accuracy[i, j]:.2f}', ha='center', va='center', color='white')
    #plt.gca().set_xticks(np.arange(-0.5, Accuracy.shape[1], 1), minor=True)
    #plt.gca().set_yticks(np.arange(-0.5, Accuracy.shape[0], 1), minor=True)
    #plt.gca().grid(which="minor", color="black", linestyle='-', linewidth=1)
    #plt.gca().tick_params(which="minor", size=0)
    #plt.xlabel('L1')
    #plt.ylabel('L2')

    #plot_filename = os.path.join(folder_path_N, f'accuracyMap.png')
    #plt.tight_layout()
    #plt.savefig(plot_filename)
    #plt.close()

    #save_filename = os.path.join(folder_path_N, f'Accuracy.csv')
    #df = pd.DataFrame(Accuracy)
    #df.to_csv(save_filename, index=False)

    #plt.imshow(ValAccuracy, cmap='viridis', aspect='auto')
    #plt.colorbar()
    #plt.xticks(ticks=np.arange(len(L1_list)), labels=L1_list)
    #plt.yticks(ticks=np.arange(len(L2_list)), labels=L2_list)
    #plt.grid(color='black', linestyle='-', linewidth=1, alpha=0.5)
    #for i in range(ValAccuracy.shape[0]):  # Rows
    #    for j in range(ValAccuracy.shape[1]):  # Columns
    #        plt.text(j, i, f'{ValAccuracy[i, j]:.2f}', ha='center', va='center', color='white')
    #plt.gca().set_xticks(np.arange(-0.5, ValAccuracy.shape[1], 1), minor=True)
    #plt.gca().set_yticks(np.arange(-0.5, ValAccuracy.shape[0], 1), minor=True)
    #plt.gca().grid(which="minor", color="black", linestyle='-', linewidth=1)
    #plt.gca().tick_params(which="minor", size=0)
    #plt.xlabel('L1')
    #plt.ylabel('L2')

    #plot_filename = os.path.join(folder_path_N, f'ValaccuracyMap.png')
    #plt.tight_layout()
    #plt.savefig(plot_filename)
    #plt.close()

    #save_filename = os.path.join(folder_path_N, f'VAlAccuracy.csv')
    #df = pd.DataFrame(ValAccuracy)
    #df.to_csv(save_filename, index=False)