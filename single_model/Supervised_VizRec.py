import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import os
import ast
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import environment_vizrec

class OnlineSVM:
    def __init__(self, max_iter=1000):
        """
        Initializes the Online SVM model using SGDClassifier.
        """
        from sklearn.linear_model import SGDClassifier
        self.model = SGDClassifier(loss='hinge', max_iter=max_iter, tol=1e-3)

    def train(self, X_train, y_train):
        """
        Train the model on the provided training data.
        """
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on the test data and return the accuracy.
        """
        # do online prediction predict , partial_fit, predict, partial_fit
        all_accuracies = []
        for i in range(len(X_test)):
            y_pred = self.model.predict([X_test[i]])
            accuracy = accuracy_score([y_test[i]], y_pred)
            self.model.partial_fit([X_test[i]], [y_test[i]])
            all_accuracies.append(accuracy)

        return np.mean(all_accuracies), y_pred

    def evaluate2(self, X_test, y_test):
        """
        Evaluate the model on the test data and return the accuracy.
        """
        #one shot prediction
        y_pred = self.model.predict(X_test)
        all_accuracies = accuracy_score(y_test, y_pred)

        return all_accuracies, y_pred

def run_experiment(user_list, dataset, task):
    """
    Run the experiment using Leave-One-Out Cross-Validation (LOOCV).
    """
    result_dataframe = pd.DataFrame(columns=['User', 'Accuracy', 'Algorithm'])

    user_list = shuffle(user_list, random_state=42)
    user_list = list(user_list)  # Convert numpy array to Python list

    y_true_all = []
    y_pred_all = []

    # Leave-One-Out Cross-Validation
    for i, test_user_log in enumerate(user_list):
        train_users = user_list[:i] + user_list[i+1:]  # All users except the ith one

        # Aggregate training data
        X_train = []
        y_train = []
        for user_log in train_users:
            env = environment_vizrec.environment_vizrec()
            env.process_data(user_log, 0)
            # Convert string representations of lists to actual lists
            states = [ast.literal_eval(state) for state in env.mem_states]
            X_train.extend(states)
            y_train.extend(env.mem_action)

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # Initialize and train the OnlineSVM model
        model = OnlineSVM()
        model.train(X_train, y_train)

        # Test on the left-out user
        user_name = os.path.basename(test_user_log).replace('_log.csv', '')
        env.process_data(test_user_log, 0)

        # Convert string representations of lists to actual lists for test data
        X_test = np.array([ast.literal_eval(state) for state in env.mem_states])
        y_test = np.array(env.mem_action)

        # Evaluate the model on the test data for this user
        accuracy, y_pred = model.evaluate(X_test, y_test)

        # Store results
        result_dataframe = pd.concat([result_dataframe, pd.DataFrame({
            'User': [user_name],
            'Accuracy': [accuracy],
            'Algorithm': ['OnlineSVM']
        })], ignore_index=True)

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

        print(f"User {user_name} - Accuracy: {accuracy:.2f}")

    # Save results to CSV
    result_dataframe.to_csv(f"Experiments_Folder/VizRec/{dataset}/{task}/SVM.csv", index=False)
    plot_predictions(y_true_all, y_pred_all, task, dataset)

def plot_predictions(y_true_all, y_pred_all, task, dataset):
    """
    Plot the predictions for all users.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(y_true_all, label='True', alpha=0.6)
    plt.plot(y_pred_all, label='Predicted', alpha=0.6)
    plt.legend()
    plt.title(f"True vs Predicted Actions for Task {task}")
    plt.savefig(f"Experiments_Folder/VizRec/{dataset}/{task}/predictions_plot.png")
    plt.close()

def get_average_accuracy():
    """
    Get the average accuracy of the Online SVM model.
    """
    #for each dataset and task, read the results and calculate the average accuracy
    datasets = ['movies', 'birdstrikes']
    tasks = ['p1', 'p2', 'p3', 'p4']
    results = []
    for dataset in datasets:
        for task in tasks:
            result_df = pd.read_csv(f"Experiments_Folder/VizRec/{dataset}/{task}/SVM.csv")
            print(f"Dataset: {dataset}, Task: {task}, Average Accuracy: {result_df['Accuracy'].mean()}")
            results.append(result_df['Accuracy'].mean())
    print(f"Overall Average Accuracy: {np.mean(results)}")


if __name__ == "__main__":
    datasets = ['movies', 'birdstrikes']
    tasks = ['p1', 'p2', 'p3', 'p4']
    for dataset in datasets:
        for task in tasks:
            env = environment_vizrec.environment_vizrec()
            user_list = env.get_user_list(dataset, task)
            run_experiment(user_list, dataset, task)
            print(f"Done with {dataset} {task}")
    get_average_accuracy()
