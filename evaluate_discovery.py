import json
import random
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, adjusted_rand_score

class EpistemicEvaluator:
    def __init__(self, truth_file="ground_truth_G.json", eval_file="evaluation_set_H.json"):
        """
        Loads the hidden ground truth and the evaluation set.
        """
        try:
            with open(truth_file, 'r') as f:
                self.truth = json.load(f)
            with open(eval_file, 'r') as f:
                self.eval_data = json.load(f)
        except FileNotFoundError:
            print(f"Error: Could not find {truth_file} or {eval_file}. Run generate_world.py first.")
            exit()

        self.node_classes = self.truth['node_classes']
        self.interaction_matrix = np.array(self.truth['interaction_matrix'])
        
        # Prepare lists for scikit-learn
        self.y_true = []
        self.test_pairs = []
        
        # 1 = Edge Exists, 0 = Edge Does Not Exist
        for u, v in self.eval_data['test_positives']:
            self.test_pairs.append((u, v))
            self.y_true.append(1)
            
        for u, v in self.eval_data['test_negatives']:
            self.test_pairs.append((u, v))
            self.y_true.append(0)

    def get_baseline_performance(self):
        """
        M0: The Baseline.
        Simulates an agent that has NOT discovered the rule and guesses 
        based on the global density of connections (Prior Probability).
        """
        # Calculate rough density from the interaction matrix to simulate prior knowledge
        avg_density = np.mean(self.interaction_matrix)
        
        # Generate random predictions weighted by density
        # (i.e., if 30% of world connects, guess "Yes" 30% of the time)
        y_pred_baseline = [1 if random.random() < avg_density else 0 for _ in range(len(self.y_true))]
        
        return {
            "accuracy": accuracy_score(self.y_true, y_pred_baseline),
            "f1": f1_score(self.y_true, y_pred_baseline)
        }

    def evaluate_llm_predictions(self, llm_predictions):
        """
        M1: The LLM.
        Calculates metrics based on the LLM's specific yes/no predictions
        for the hidden evaluation pairs.
        
        :param llm_predictions: A list of 0s and 1s corresponding to self.test_pairs
        """
        if len(llm_predictions) != len(self.y_true):
            raise ValueError(f"LLM provided {len(llm_predictions)} predictions, expected {len(self.y_true)}.")

        acc = accuracy_score(self.y_true, llm_predictions)
        f1 = f1_score(self.y_true, llm_predictions)
        
        return {"accuracy": acc, "f1": f1}

    def verify_truth_clustering(self, llm_grouped_nodes):
        """
        Verifies if the LLM correctly identified the Hidden Classes.
        This tests 'Truth' - did the LLM find the latent structure?
        
        :param llm_grouped_nodes: Dict { "Node_ID": Predicted_Cluster_ID }
        """
        # Align ground truth lists with LLM prediction keys
        nodes = list(llm_grouped_nodes.keys())
        true_labels = [self.node_classes.get(n) for n in nodes]
        pred_labels = [llm_grouped_nodes[n] for n in nodes]
        
        # Filter out nodes that might not exist in ground truth (hallucinations)
        valid_indices = [i for i, label in enumerate(true_labels) if label is not None]
        true_labels = [true_labels[i] for i in valid_indices]
        pred_labels = [pred_labels[i] for i in valid_indices]

        # Adjusted Rand Index (ARI):
        # 0.0 = Random grouping
        # 1.0 = Perfect match (even if cluster IDs 0,1,2 are swapped to A,B,C)
        ari = adjusted_rand_score(true_labels, pred_labels)
        return ari

    def calculate_epistemic_gain(self, baseline_metrics, llm_metrics):
        """Calculates Delta (Epistemic Gain)."""
        delta_acc = llm_metrics['accuracy'] - baseline_metrics['accuracy']
        delta_f1 = llm_metrics['f1'] - baseline_metrics['f1']
        
        return delta_acc, delta_f1

# --- Simulation of the Experiment ---
if __name__ == "__main__":
    evaluator = EpistemicEvaluator()
    
    print(f"--- Evaluating {len(evaluator.test_pairs)} Hidden Facts (H) ---")
    
    # 1. Run M0 (Baseline)
    baseline_res = evaluator.get_baseline_performance()
    print(f"M0 (Baseline) Accuracy: {baseline_res['accuracy']:.2f}")

    # 2. Simulate M1 (LLM Predictions)
    # NOTE: In a real experiment, you would parse the LLM's "Yes/No" answers here.
    # Here, we simulate a "Smart" LLM that discovered the rule (95% accurate)
    # and a "Hallucinating" LLM (random guessing).
    
    print("\n--- Simulation: Smart LLM (Discovered the Rule) ---")
    # Simulate predictions that match the ground truth with 95% noise flip
    simulated_smart_preds = []
    for truth in evaluator.y_true:
        if random.random() > 0.95: 
            simulated_smart_preds.append(1 - truth) # Error
        else:
            simulated_smart_preds.append(truth) # Correct
            
    llm_res = evaluator.evaluate_llm_predictions(simulated_smart_preds)
    delta_acc, delta_f1 = evaluator.calculate_epistemic_gain(baseline_res, llm_res)
    
    print(f"M1 (LLM) Accuracy:      {llm_res['accuracy']:.2f}")
    print(f"Epistemic Gain (Delta): {delta_acc:.2f}")
    
    if delta_acc > 0.2: # Threshold tau
        print("RESULT: Genuine Knowledge Discovery Detected!")
    else:
        print("RESULT: No significant discovery.")

    # 3. Verify Truth (Clustering)
    # Simulate the LLM saying: "I think Entity_A and Entity_B are the same type."
    print("\n--- Verifying Truth (Clustering) ---")
    
    # Create a mock clustering that matches the truth perfectly
    mock_llm_clustering = evaluator.node_classes.copy() 
    ari_score = evaluator.verify_truth_clustering(mock_llm_clustering)
    
    print(f"Cluster Match Score (ARI): {ari_score:.2f} (1.0 is perfect)")
    if ari_score > 0.8:
        print("RESULT: The LLM correctly induced the Hidden Classes.")