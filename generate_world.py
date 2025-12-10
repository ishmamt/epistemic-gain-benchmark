import networkx as nx
import numpy as np
import random
import json
import uuid

class StochasticWorldGenerator:
    def __init__(self, num_nodes=50, num_classes=3, density=0.3, seed=42):
        """
        Initializes the world generator.
        
        :param num_nodes: Total entities in the world.
        :param num_classes: Number of latent hidden classes (e.g., Fire, Water, Earth).
        :param density: General density of connections.
        :param seed: Random seed for reproducibility (Scientific Rigor).
        """
        self.num_nodes = num_nodes
        self.num_classes = num_classes
        self.seed = seed
        self.density = density
        
        np.random.seed(seed)
        random.seed(seed)
        
        # 1. Define the Ground Truth (G) - The Hidden Laws of Physics
        # Interaction Matrix: prob of Class i -> Class j
        # We create a structured matrix so a rule actually EXISTS to be discovered.
        # Example: A strict hierarchy (0->1, 1->2, 2->0) like Rock-Paper-Scissors.
        self.interaction_matrix = np.zeros((num_classes, num_classes))
        
        # Creating a specific cyclic rule: Class i interacts heavily with (i+1)%k
        for i in range(num_classes):
            target = (i + 1) % num_classes
            self.interaction_matrix[i][target] = 0.85  # High probability (The Rule)
            
            # Add some noise (random connections) to make it stochastic, not deterministic
            for j in range(num_classes):
                if j != target:
                    self.interaction_matrix[i][j] = 0.05 # Low probability (Noise)

        self.graph = nx.DiGraph()
        self.nodes = []
        self.node_classes = {}

    def generate_graph(self):
        """Generates the graph based on the SBM probabilities."""
        # 2. Create Nodes and assign Hidden Classes
        for i in range(self.num_nodes):
            # Generate a unique, non-semantic ID (e.g., "X-92", "B-14")
            # to prevent the LLM from using prior training data.
            node_id = f"Entity_{uuid.uuid4().hex[:4].upper()}"
            hidden_class = random.randint(0, self.num_classes - 1)
            
            self.nodes.append(node_id)
            self.node_classes[node_id] = hidden_class
            self.graph.add_node(node_id, hidden_class=hidden_class)

        # 3. Generate Edges based on Interaction Matrix
        for u in self.nodes:
            for v in self.nodes:
                if u == v: continue
                
                class_u = self.node_classes[u]
                class_v = self.node_classes[v]
                
                # Probability of interaction determined by the hidden matrix
                prob = self.interaction_matrix[class_u][class_v]
                
                if random.random() < prob:
                    self.graph.add_edge(u, v)

    def create_experiment_data(self, test_ratio=0.2):
        """
        Splits the graph into Observed (O) and Hidden (H).
        Returns dictionaries ready for JSON export.
        """
        all_edges = list(self.graph.edges())
        random.shuffle(all_edges)
        
        split_idx = int(len(all_edges) * (1 - test_ratio))
        
        # O: The Context provided to the LLM
        observed_edges = all_edges[:split_idx]
        
        # H: The Hidden Facts for Epistemic Value calculation
        hidden_edges = all_edges[split_idx:]
        
        # --- Prompt Construction (The Input) ---
        prompt_lines = ["Here is a log of observed interactions between different entities in a closed system."]
        for u, v in observed_edges:
            prompt_lines.append(f"- {u} triggers {v}")
        
        prompt_lines.append("\nTASK: Based on these observations, propose a general rule governing how these entities interact. Then, predict likely future interactions.")

        # --- Truth Construction (For Evaluation) ---
        # We export the class map and the matrix so you can verify Novelty/Truth.
        ground_truth = {
            "interaction_matrix": self.interaction_matrix.tolist(),
            "node_classes": self.node_classes,
            "rule_description": "Cyclic hierarchy: Class 0->1, 1->2, 2->0 (approx 85% chance)."
        }

        # --- Epistemic Test Set (For Delta Calculation) ---
        # We need positive samples (actual hidden edges) and negative samples (non-edges)
        # to calculate Accuracy/AUC for M0 vs M1.
        
        # Get actual edges (Positive samples)
        test_positives = hidden_edges
        
        # Get non-existent edges (Negative samples)
        test_negatives = []
        while len(test_negatives) < len(test_positives):
            u, v = random.choice(self.nodes), random.choice(self.nodes)
            if not self.graph.has_edge(u, v) and u != v:
                test_negatives.append((u, v))

        evaluation_set = {
            "test_positives": test_positives,
            "test_negatives": test_negatives
        }

        return "\n".join(prompt_lines), ground_truth, evaluation_set

# --- Execution ---
if __name__ == "__main__":
    # Settings
    N_NODES = 30
    N_CLASSES = 3
    
    # Run Generation
    gen = StochasticWorldGenerator(num_nodes=N_NODES, num_classes=N_CLASSES)
    gen.generate_graph()
    
    prompt, truth, eval_data = gen.create_experiment_data()
    
    # Save Files
    with open("experiment_prompt_O.txt", "w") as f:
        f.write(prompt)
        
    with open("ground_truth_G.json", "w") as f:
        json.dump(truth, f, indent=4)
        
    with open("evaluation_set_H.json", "w") as f:
        json.dump(eval_data, f, indent=4)

    print(f"Experiment generated.")
    print(f"Nodes: {N_NODES}, Classes: {N_CLASSES}")
    print(f"Files created: experiment_prompt_O.txt, ground_truth_G.json, evaluation_set_H.json")
    
    # Preview logic check
    print("\n--- Interaction Matrix (The Hidden Rule) ---")
    print(np.array(truth['interaction_matrix']))