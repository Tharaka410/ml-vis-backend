from sklearn.preprocessing import StandardScaler
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from sklearn.cluster import DBSCAN, KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression, Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal, Optional
from sklearn.datasets import load_iris, load_wine, load_digits, load_breast_cancer, make_classification
import numpy as np
from minisom import MiniSom
from collections import Counter
from gain_ratio import GainRatioDecisionTree # Keep this import
import random
from typing import List, Dict, Callable, Literal, Optional
from sklearn.linear_model import Perceptron as SklearnPerceptron # Alias to avoid name conflict
from sklearn.neural_network import MLPClassifier # Import MLPClassifier
from sklearn.preprocessing import StandardScaler # For sklearn Perceptron if needed


DATASETS = {
    "iris": datasets.load_iris,
    "wine": datasets.load_wine,
    "digits": datasets.load_digits,
}

class TreeParams(BaseModel):
    max_depth: int
    min_samples_split: int
    criterion: Literal["gini", "entropy", "gain_ratio"] # Keep 'gain_ratio' for standalone tree

class TreeRequest(BaseModel):
    dataset: Literal["iris", "wine", "digits"]
    params: TreeParams

def get_dataset(name: str):
    if name in DATASETS:
        return DATASETS[name]()
    else:
        raise HTTPException(status_code=400, detail="Unsupported dataset")

def build_sklearn_tree(dataset_obj, criterion: str, max_depth: int):
    clf = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth)
    clf.fit(dataset_obj.data, dataset_obj.target)

    tree_ = clf.tree_

    def recurse(node: int):
        if tree_.feature[node] == -2:
            # leaf node
            return {
                "feature": None,
                "threshold": None,
                "left": None,
                "right": None,
                "value": int(np.argmax(tree_.value[node])),
                "impurity": float(tree_.impurity[node]),
            }
        else:
            return {
                "feature": int(tree_.feature[node]),
                "threshold": float(tree_.threshold[node]),
                "left": recurse(tree_.children_left[node]),
                "right": recurse(tree_.children_right[node]),
                "value": None,
                "impurity": float(tree_.impurity[node]),
            }

    return recurse(0)

def sklearn_tree_to_dict(tree_model, feature_names, class_names):
    tree_ = tree_model.tree_
    feature_name = [
        feature_names[i] if i != -2 else "leaf"
        for i in tree_.feature
    ]

    def recurse(node_id):
        if tree_.feature[node_id] == -2: # Leaf node
            return {
                "feature": None,
                "threshold": None,
                "impurity": tree_.impurity[node_id],
                "left": None,
                "right": None,
                "value": int(np.argmax(tree_.value[node_id])),
                "samples": int(tree_.n_node_samples[node_id])
            }
        else: # Internal node
            return {
                "feature": feature_name[node_id],
                "threshold": tree_.threshold[node_id],
                "impurity": tree_.impurity[node_id],
                "left": recurse(tree_.children_left[node_id]),
                "right": recurse(tree_.children_right[node_id]),
                "value": None,
                "samples": int(tree_.n_node_samples[node_id])
            }

    return recurse(0)


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "https://ml-vis-lbhl.onrender.com","https://ml-vis-frontend.onrender.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Point(BaseModel):
    x: float
    y: float

class DBSCANRequest(BaseModel):
    points: List[Point]
    epsilon: float
    minPoints: int

class KMeansRequest(BaseModel):
    points: List[Point]
    clusters: int

class SupervisedRequest(BaseModel):
    X: List[List[float]]
    y: List[int]

class PredictionResponse(BaseModel):
    predictions: List[int]

class DBSCANResponse(BaseModel):
    labels: List[int]

class KMeansResponse(BaseModel):
    labels: List[int]
    centers: List[List[float]]

class DatasetParams(BaseModel):
    dataset: Literal["iris", "wine", "digits"]
    max_depth: int = 3
    min_samples_split: int = 2
    criterion: Literal["gini", "entropy", "gain_ratio"] # Keep 'gain_ratio' for standalone tree
    split_ratio: float = 0.3

class Dataset(BaseModel):
    features: List[List[float]]
    labels: List[int]

class ForestParams(BaseModel):
    n_trees: int
    subsample_ratio: float
    feature_subset: float
    max_depth: Optional[int]
    min_samples_split: int
    criterion: Literal["gini", "entropy"] # Removed 'gain_ratio' for Random Forest

class ForestRequest(BaseModel):
    dataset: Literal["iris", "wine", "digits"]
    params: ForestParams
    
class PredictionRequest(BaseModel):
    dataset: Literal["iris", "wine", "digits"]
    params: ForestParams # This also restricts criterion to "gini" or "entropy" for prediction
    record: List[float] # The input record for prediction
    
class LinearRegressionRequest(BaseModel):
    X: List[List[float]]
    y: List[float]
    
class LinearRegressionHistoryRequest(BaseModel):
    X: List[List[float]]
    y: List[float]
    iterations: int

class SOMTrainRequest(BaseModel):
    data: List[List[float]] # Input data points (e.g., [x, y, z] for 3D)
    grid_size_x: int
    grid_size_y: int
    learning_rate: float
    iterations: int
    sigma: float
    initial_weights: Optional[List[List[List[float]]]] = None


class SOMNode(BaseModel):
    x: float
    y: float
    z: float # For 3D visualization
    i: int
    j: int

class SOMTrainResponse(BaseModel):
    final_weights: List[List[List[float]]] # [grid_x][grid_y][features]
    history: List[List[List[List[float]]]] # List of weights for each iteration
    bmu_history: List[List[float]] # History of BMU input coordinates [x, y, z]
    
    

class LogisticRegressionTrainRequest(BaseModel):
    X: List[List[float]]
    y: List[int]
    learning_rate: float
    iterations: int


@app.post("/dbscan", response_model=DBSCANResponse)
def run_dbscan(req: DBSCANRequest):
    data = np.array([[p.x, p.y] for p in req.points])
    model = DBSCAN(eps=req.epsilon, min_samples=req.minPoints)
    labels = model.fit_predict(data)
    return {"labels": labels.tolist()}

@app.post("/kmeans", response_model=KMeansResponse)
def run_kmeans(req: KMeansRequest):
    data = np.array([[p.x, p.y] for p in req.points])
    model = KMeans(n_clusters=req.clusters, random_state=0, n_init=10)
    labels = model.fit_predict(data)
    centers = model.cluster_centers_
    return {"labels": labels.tolist(), "centers": centers.tolist()}

@app.post("/decision-tree", response_model=PredictionResponse)
def train_tree(params: TreeParams):
    if params.dataset not in DATASETS:
        raise HTTPException(status_code=400, detail="Invalid dataset selected.")


    data = DATASETS[params.dataset]()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params.split_ratio, random_state=42
    )

    if params.criterion in ["gini", "entropy"]:
        clf = DecisionTreeClassifier(
            max_depth=params.max_depth,
            min_samples_split=params.min_samples_split,
            criterion=params.criterion
        )
        clf.fit(X_train, y_train)

        return {
            "feature_names": data.feature_names,
            "classes": data.target_names.tolist() if hasattr(data, "target_names") else list(set(y)),
            "tree_depth": clf.get_depth(),
            "n_leaves": clf.get_n_leaves(),
            "train_score": clf.score(X_train, y_train),
            "test_score": clf.score(X_test, y_test),
        }

    elif params.criterion == "gain_ratio":
        clf = GainRatioDecisionTree(
            max_depth=params.max_depth,
            min_samples_split=params.min_samples_split
        )
        clf.fit(X_train, y_train)
        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)

        train_score = np.mean(y_pred_train == y_train)
        test_score = np.mean(y_pred_test == y_test)

        return {
            "feature_names": data.feature_names,
            "classes": data.target_names.tolist() if hasattr(data, "target_names") else list(set(y)),
            "tree_depth": params.max_depth,  # Estimating tree depth since custom model doesn't compute exact
            "n_leaves": None,
        }


@app.post("/build_tree")
def build_tree(req: TreeRequest):
    data = get_dataset(req.dataset)
    criterion = req.params.criterion.lower()

    if criterion in ["gini", "entropy"]:
        tree_json = build_sklearn_tree(data, criterion, req.params.max_depth)
    elif criterion == "gain_ratio":
        clf = GainRatioDecisionTree(
            max_depth=req.params.max_depth,
            min_samples_split=req.params.min_samples_split,
        )
        clf.fit(data.data, data.target)
        tree_json = clf.to_dict()
    else:
        raise HTTPException(status_code=400, detail="Unsupported criterion")

    return {
    "tree": tree_json,
    "feature_names": data.feature_names,
}




@app.get("/logistic-regression/data") # Changed to GET
async def get_logistic_regression_data():
    # Generate a synthetic dataset for binary classification
    X, y = make_classification(
        n_samples=200,          # Number of samples
        n_features=2,           # Only 2 features for 2D visualization
        n_informative=2,        # Both features are informative
        n_redundant=0,          # No redundant features
        n_repeated=0,           # No repeated features
        n_classes=2,            # Binary classification
        n_clusters_per_class=1, # One cluster per class
        flip_y=0.1,             # 10% of samples have their class flipped (adds noise/overlap)
        random_state=42         # For reproducibility
    )
    
    feature_names = ["Synthetic Feature 1", "Synthetic Feature 2"]

    # Standardize features (important for logistic regression convergence)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).tolist()

    return {
        "X": X_scaled,
        "y": y.tolist(),
        "feature_names": feature_names,
    }

@app.post("/logistic-regression/train-history")
async def train_logistic_regression(req: LogisticRegressionTrainRequest):
    X = np.array(req.X)
    y = np.array(req.y)

    num_samples, num_features = X.shape

    # Initialize weights and bias
    weights = np.zeros(num_features)
    bias = 0.0

    weights_history = []
    loss_history = []

    for i in range(req.iterations):
        # Calculate linear model output: z = X @ weights + bias
        linear_model = np.dot(X, weights) + bias
        
        # Apply sigmoid function to get probabilities
        # Ensure numerical stability for exp(-z)
        predictions = 1 / (1 + np.exp(-np.clip(linear_model, -500, 500))) # Clip to prevent overflow

        # Calculate the error for gradient descent
        error = predictions - y

        # Update weights and bias using gradient descent
        dw = (1 / num_samples) * np.dot(X.T, error)
        db = (1 / num_samples) * np.sum(error)

        weights -= req.learning_rate * dw
        bias -= req.learning_rate * db

        # Calculate Binary Cross-Entropy Loss
        # Add a small epsilon to probabilities to avoid log(0) or log(1-0) issues
        epsilon = 1e-10
        current_loss = -np.mean(y * np.log(predictions + epsilon) + (1 - y) * np.log(1 - predictions + epsilon))
        
        # Store history
        weights_history.append(weights.tolist() + [bias])
        loss_history.append(current_loss)

    # Calculate final predictions and loss for summary
    final_linear_model = np.dot(X, weights) + bias
    final_predictions_prob = 1 / (1 + np.exp(-np.clip(final_linear_model, -500, 500)))
    final_predictions = (final_predictions_prob > 0.5).astype(int).tolist()
    final_loss = -np.mean(y * np.log(final_predictions_prob + epsilon) + (1 - y) * np.log(1 - final_predictions_prob + epsilon))


    return {
        "weights_history": weights_history,
        "loss_history": loss_history,
        "final_predictions": final_predictions,
        "final_loss": final_loss,
    }





@app.post("/som_train")
async def som_train(req: SOMTrainRequest):
    data = np.array(req.data)
    input_len = data.shape[1] # Number of features (e.g., 2 for 2D, 3 for 3D data)

    # Initialize SOM
    som = MiniSom(
        x=req.grid_size_x,
        y=req.grid_size_y,
        input_len=input_len,
        sigma=req.sigma,
        learning_rate=req.learning_rate,
        random_seed=42 # For reproducibility
    )

    # If initial_weights are provided, set them
    if req.initial_weights is not None:
        try:
            som._weights = np.array(req.initial_weights)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to set initial weights: {e}")
    else:
        som.random_weights_init(data) # Initialize randomly if not provided

    history_weights = []
    bmu_input_history = []

    # Manual training loop to capture history
    for i in range(req.iterations):
        # Capture current weights at the beginning of the iteration
        history_weights.append(som.get_weights().tolist())

        # Select a random data point
        rand_idx = random.randint(0, len(data) - 1)
        datapoint = data[rand_idx]

        # Find BMU for the current datapoint
        bmu_coords = som.winner(datapoint)
        bmu_input_history.append(datapoint.tolist())

        # Update SOM weights
        som.update(datapoint, som.winner(datapoint), i, req.iterations)


    final_weights = som.get_weights().tolist()

    return {
        "final_weights": final_weights,
        "history": history_weights,
        "bmu_history": bmu_input_history
    }



@app.post("/linear-regression")
async def run_linear_regression(req: LinearRegressionRequest):
    X = np.array(req.X)
    y = np.array(req.y)

    model = LinearRegression()
    model.fit(X, y)

    predictions = model.predict(X).tolist()
    mse = np.mean((y - predictions)**2)

    return {
        "coef": model.coef_.tolist(),
        "intercept": float(model.intercept_), # Ensure intercept is a float
        "predictions": predictions,
        "final_mse": float(mse) # Return final MSE
    }


@app.post("/linear-regression-history")
async def get_linear_regression_history(req: LinearRegressionHistoryRequest):
    X = np.array(req.X)
    y = np.array(req.y)
    iterations = req.iterations

    w = 0.0
    b = 0.0
    learning_rate = 0.01

    mse_history = []
    w_history = []
    b_history = []

    for i in range(iterations):
        y_hat = w * X.flatten() + b
        error = y_hat - y

        dw = (X.flatten() * error).mean()
        db = error.mean()

        w -= learning_rate * dw
        b -= learning_rate * db

        current_mse = np.mean(error**2)
        mse_history.append(float(current_mse))
        w_history.append(float(w))
        b_history.append(float(b))

    return {
        "mse_history": mse_history,
        "w_history": w_history,
        "b_history": b_history
    }
    
    
    

@app.post("/knn", response_model=PredictionResponse)
def run_knn(req: SupervisedRequest):
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(req.X, req.y)
    return {"predictions": model.predict(req.X).tolist()}




@app.post("/svm", response_model=PredictionResponse)
def run_svm(req: SupervisedRequest):
    model = SVC()
    model.fit(req.X, req.y)
    return {"predictions": model.predict(req.X).tolist()}






def get_bootstrap_sample(X, y, ratio):
    n_samples = int(len(X) * ratio)
    indices = np.random.choice(len(X), n_samples, replace=True)
    return X[indices], y[indices]

def get_feature_subset(X, subset_ratio, all_feature_names):
    n_features = int(X.shape[1] * subset_ratio)
    if X.shape[1] == 0:
        return [], []
    num_features_to_select = max(1, n_features)
    indices = sorted(random.sample(range(X.shape[1]), num_features_to_select))
    selected_feature_names = [all_feature_names[i] for i in indices]
    return indices, selected_feature_names

def convert_to_python_list(arr_or_list):
    if isinstance(arr_or_list, np.ndarray):
        return [item.item() if isinstance(item, np.generic) else item for item in arr_or_list]
    if isinstance(arr_or_list, list):
        return [item.item() if isinstance(item, np.generic) else item for item in arr_or_list]
    return arr_or_list


@app.post("/build_forest")
async def build_forest(req: ForestRequest):
    dataset_obj = get_dataset(req.dataset)
    X, y = dataset_obj.data, dataset_obj.target
    feature_names = dataset_obj.feature_names if hasattr(dataset_obj, "feature_names") else [f"feature_{i}" for i in range(X.shape[1])]

    trees_data = []
    for _ in range(req.params.n_trees):
        # Bootstrap sampling
        X_sample, y_sample = get_bootstrap_sample(X, y, req.params.subsample_ratio)

        # Feature subsetting
        feature_indices, selected_feature_names = get_feature_subset(
            X_sample, req.params.feature_subset, feature_names
        )
        X_selected_for_training = X_sample[:, feature_indices]

        # Build tree using scikit-learn's DecisionTreeClassifier (only gini/entropy allowed for Random Forest)
        clf = DecisionTreeClassifier(
            max_depth=req.params.max_depth,
            min_samples_split=req.params.min_samples_split,
            criterion=req.params.criterion,
            random_state=None # Keep random_state None for randomness in forest
        )
        clf.fit(X_selected_for_training, y_sample)

        trees_data.append({
            "tree": sklearn_tree_to_dict(clf, selected_feature_names, dataset_obj.target_names),
            "feature_indices": feature_indices # Removed .tolist() as feature_indices is already a list
        })

    return {
        "trees": trees_data,
        "target_names": convert_to_python_list(dataset_obj.target_names) if hasattr(dataset_obj, "target_names") else sorted(list(set(y.tolist()))) # Ensure Python types
    }

@app.post("/predict_forest")
async def predict_forest(req: PredictionRequest):
    dataset_obj = get_dataset(req.dataset)
    X, y = dataset_obj.data, dataset_obj.target
    feature_names = dataset_obj.feature_names if hasattr(dataset_obj, "feature_names") else [f"feature_{i}" for i in range(X.shape[1])]

    votes = []
    for _ in range(req.params.n_trees):
        # Bootstrap sampling (same as build_forest to get consistent feature_indices)
        X_sample, y_sample = get_bootstrap_sample(X, y, req.params.subsample_ratio)
        feature_indices, selected_feature_names = get_feature_subset(
            X_sample, req.params.feature_subset, feature_names
        )
        X_selected_for_training = X_sample[:, feature_indices]

        # Build tree using scikit-learn's DecisionTreeClassifier (only gini/entropy allowed for Random Forest)
        clf = DecisionTreeClassifier(
            max_depth=req.params.max_depth,
            min_samples_split=req.params.min_samples_split,
            criterion=req.params.criterion,
            random_state=None # Keep random_state None for randomness in forest
        )
        clf.fit(X_selected_for_training, y_sample)

        # Select features from the input record based on the current tree's subset
        input_record_for_tree = [
            req.record[idx] for idx in feature_indices
        ]

        pred = clf.predict([input_record_for_tree])[0]
        votes.append(int(pred)) # Ensure individual votes are ints

    vote_counts = Counter(votes)
    majority_vote = int(vote_counts.most_common(1)[0][0]) # Ensure majority_vote is int

    return {
        "individual_votes": votes,
        "majority_vote": majority_vote,
        "class_names": convert_to_python_list(dataset_obj.target_names) if hasattr(dataset_obj, "target_names") else sorted(list(set(y.tolist()))) # Ensure Python types
    }
    
    


# --- Activation Functions for CustomPerceptron ---
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(y):
    return y * (1 - y)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(y):
    return np.where(y > 0, 1, 0)

def identity(x):
    return x

def identity_derivative(y):
    return np.ones_like(y)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def softmax_derivative(y):
    return np.ones_like(y) # Placeholder, actual derivative depends on loss


ACTIVATION_FUNCTIONS: Dict[str, Dict[str, Callable]] = {
    "sigmoid": {"func": sigmoid, "derivative": sigmoid_derivative},
    "relu": {"func": relu, "derivative": relu_derivative},
    "identity": {"func": identity, "derivative": identity_derivative},
    "softmax": {"func": softmax, "derivative": softmax_derivative}
}

# Mapping for sklearn activation names
SKLEARN_ACTIVATION_MAP = {
    "identity": "identity",
    "sigmoid": "logistic",
    "relu": "relu",
    "softmax": "identity" # MLPClassifier handles softmax activation implicitly for multi-class
}


# --- CustomPerceptron Class (remains largely the same) ---
class CustomPerceptron:
    def __init__(self, input_size: Optional[int] = None, hidden_size: Optional[int] = None,
                 output_nodes: Optional[int] = None, hidden_layers: int = 0,
                 activation_name: str = "sigmoid", output_activation_name: str = "sigmoid",
                 layers_data: Optional[List[Dict[str, List[List[float]]]]] = None):
        """
        Initializes a custom multi-layer perceptron.
        Args:
            input_size: Number of input features.
            hidden_size: Number of neurons in each hidden layer.
            output_nodes: Number of output neurons.
            hidden_layers: Number of hidden layers.
            activation_name: Name of the activation function for hidden layers.
            output_activation_name: Name of the activation function for the output layer.
            layers_data: Optional pre-existing layer weights/biases for loading.
        """
        self.activation_func = ACTIVATION_FUNCTIONS[activation_name]["func"]
        self.activation_derivative = ACTIVATION_FUNCTIONS[activation_name]["derivative"]
        self.output_activation_func = ACTIVATION_FUNCTIONS[output_activation_name]["func"]
        self.output_activation_derivative = ACTIVATION_FUNCTIONS[output_activation_name]["derivative"]

        self.layers: List[Dict[str, np.ndarray]] = [] # Stores weights and biases

        if layers_data:
            # Load existing network from layers_data
            for layer_model in layers_data:
                self.layers.append({
                    "weights": np.array(layer_model['weights']),
                    "biases": np.array(layer_model['biases'])
                })
        elif input_size is not None and hidden_size is not None and output_nodes is not None:
            # Initialize new network
            if hidden_layers == 0:
                # Direct input to output layer
                self.layers.append({
                    "weights": np.random.randn(output_nodes, input_size) * 0.01,
                    "biases": np.zeros(output_nodes)
                })
            else:
                # Input to first hidden layer
                self.layers.append({
                    "weights": np.random.randn(hidden_size, input_size) * 0.01,
                    "biases": np.zeros(hidden_size)
                })

                # Additional hidden layers
                for _ in range(hidden_layers - 1):
                    self.layers.append({
                        "weights": np.random.randn(hidden_size, hidden_size) * 0.01,
                        "biases": np.zeros(hidden_size)
                    })

                # Last hidden layer to output layer
                self.layers.append({
                    "weights": np.random.randn(output_nodes, hidden_size) * 0.01,
                    "biases": np.zeros(output_nodes)
                })
        else:
            raise ValueError("Either 'layers_data' or all of 'input_size', 'hidden_size', 'output_nodes' must be provided for initialization.")


    def forward_pass(self, input_data: List[float]) -> List[np.ndarray]:
        activations_list = [np.array(input_data)] # First element is input
        current_activation = np.array(input_data)

        for i, layer in enumerate(self.layers):
            weights = layer["weights"]
            biases = layer["biases"]
            net_input = np.dot(weights, current_activation) + biases

            if i == len(self.layers) - 1: # Output layer
                current_activation = self.output_activation_func(net_input)
            else: # Hidden layer
                current_activation = self.activation_func(net_input)
            activations_list.append(current_activation)
        return activations_list

    def backward_pass(self, input_data: List[float], target: List[float],
                      activations_list_raw: List[np.ndarray], learning_rate: float) -> List[np.ndarray]:
        targets = np.array(target)
        output = activations_list_raw[-1]
        error = output - targets

        if self.output_activation_derivative == softmax_derivative: # Special handling for softmax with common loss
             output_delta = error
        else:
            output_delta = error * self.output_activation_derivative(output)

        deltas = [output_delta]

        for i in reversed(range(len(self.layers) - 1)):
            current_layer_activation = activations_list_raw[i + 1]
            prev_layer_activation = activations_list_raw[i]
            next_layer_weights = self.layers[i+1]["weights"]
            next_layer_delta = deltas[0]

            error_propagated = np.dot(next_layer_weights.T, next_layer_delta)

            delta = error_propagated * self.activation_derivative(current_layer_activation)
            deltas.insert(0, delta)

        for i, layer in enumerate(self.layers):
            current_input = activations_list_raw[i]
            delta = deltas[i]

            if current_input.ndim == 1:
                current_input = current_input.reshape(-1, 1)

            if delta.ndim == 1:
                delta = delta.reshape(-1, 1)

            if delta.shape[1] != 1 or current_input.shape[0] != 1:
                weight_update = np.dot(delta, current_input.T)
            else:
                weight_update = np.outer(delta, current_input)

            layer["weights"] -= learning_rate * weight_update
            layer["biases"] -= learning_rate * delta.flatten()

        return deltas

# --- Sklearn Perceptron/MLPClassifier Wrapper ---
class SklearnPerceptronWrapper:
    def __init__(self, input_size: int, output_nodes: int, hidden_size: int = 0, hidden_layers: int = 0,
                 activation_name: str = "relu", output_activation_name: str = "identity",
                 learning_rate: float = 0.001, random_state: int = None,
                 layers_data: Optional[List[Dict[str, List[List[float]]]]] = None):

        self.input_size = input_size
        self.output_nodes = output_nodes
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.output_activation_name = output_activation_name
        self.fitted = False
        self.scaler = StandardScaler()
        self.model = None

        if hidden_layers == 0:
            self.model = SklearnPerceptron(
                eta0=learning_rate,
                fit_intercept=True,
                shuffle=False,
                random_state=random_state,
                max_iter=1,
                tol=None
            )
            if layers_data and len(layers_data) == 1:
                # Manually set coef_ and intercept_ if pre-trained data provided
                self.model.coef_ = np.array(layers_data[0]['weights'])
                self.model.intercept_ = np.array(layers_data[0]['biases'])
                self.fitted = True # Mark as fitted if weights are provided
        else:
            hidden_layer_sizes = tuple([hidden_size] * hidden_layers)
            
            # Map activation name to sklearn's
            sklearn_activation = SKLEARN_ACTIVATION_MAP.get(activation_name, "relu")
            if output_activation_name == "softmax":
                self.model = MLPClassifier(
                    hidden_layer_sizes=hidden_layer_sizes,
                    activation=sklearn_activation,
                    solver='sgd', # Stochastic Gradient Descent
                    learning_rate_init=learning_rate,
                    max_iter=1, # Train for one iteration per call
                    shuffle=False,
                    random_state=random_state,
                    tol=None, # Disable tolerance for single-step training
                    warm_start=True, # To allow incremental fitting
                    alpha=0.0001, 
                )
            else:
                 self.model = MLPClassifier(
                    hidden_layer_sizes=hidden_layer_sizes,
                    activation=sklearn_activation,
                    solver='sgd', # Stochastic Gradient Descent
                    learning_rate_init=learning_rate,
                    max_iter=1, # Train for one iteration per call
                    shuffle=False,
                    random_state=random_state,
                    tol=None, # Disable tolerance for single-step training
                    warm_start=True, # To allow incremental fitting
                    alpha=0.0001, 
                )


            if layers_data:
                # Load pre-trained weights and biases for MLPClassifier
                self.model.coefs_ = [np.array(layer['weights']) for layer in layers_data]
                self.model.intercepts_ = [np.array(layer['biases']) for layer in layers_data]
                self.model.n_layers_ = len(layers_data) + 1 # Input layer + num_layers
                self.model.n_outputs_ = self.output_nodes # Set number of outputs
                self.model.out_activation_ = SKLEARN_ACTIVATION_MAP.get(output_activation_name, "identity") # For MLP
                self.fitted = True # Mark as fitted

    def _apply_output_activation_for_display(self, net_input: np.ndarray, activation_name: str) -> np.ndarray:
        """Applies specified activation to the raw output for consistent display."""
        func = ACTIVATION_FUNCTIONS.get(activation_name, {}).get("func", identity)
        return func(net_input)

    def forward_pass(self, input_data: List[float], output_activation_name: str = "identity") -> List[np.ndarray]:
        X = np.array(input_data).reshape(1, -1)

        # Handle scaling consistently
        if self.fitted:
            X_processed = self.scaler.transform(X)
        else:
            # If not fitted, we might not have a scaler yet.
            # If layers_data was provided and it's an MLPClassifier, it can predict.
            # If it's a Perceptron and not fitted, its coef_ and intercept_ will be None.
            X_processed = X

        activations_list = [X.flatten()] # First element is input

        if not self.fitted and (self.model is None or not hasattr(self.model, 'coefs_') or self.model.coefs_ is None):
            # Before first fit or if no pre-trained weights, return dummy output
            initial_output = np.random.rand(self.output_nodes) * 0.01
            activations_list.append(initial_output)
            return activations_list

        if isinstance(self.model, SklearnPerceptron):
            # For SklearnPerceptron, decision_function gives raw scores
            net_input = self.model.decision_function(X_processed)
            if self.output_nodes == 1:
                # Sklearn Perceptron is inherently binary, returns 1D array for single class
                output = self._apply_output_activation_for_display(net_input.flatten(), output_activation_name)
            else:
                 # Sklearn Perceptron is not designed for multi-output directly like this.
                 # This branch might need more thought if the user truly wants multi-output Perceptron.
                 # For now, let's treat it as if decision_function can return multiple scores.
                 output = self._apply_output_activation_for_display(net_input.flatten(), output_activation_name)

            activations_list.append(output)
            return activations_list

        elif isinstance(self.model, MLPClassifier):
            # For MLPClassifier, we need to manually compute activations for each layer
            # because .predict_proba or .predict only give the final output.
            # ._forward_pass is internal, so we replicate logic for visualization.
            current_activation = X_processed.copy()
            activations_list_raw = [X.flatten()] # Store original input for visualization if needed

            # Iterate through layers to get intermediate activations
            for i in range(len(self.model.coefs_)):
                net_input = np.dot(current_activation, self.model.coefs_[i]) + self.model.intercepts_[i]
                
                # Apply activation function for hidden layers or output layer
                if i < len(self.model.coefs_) - 1: # Hidden layers
                    current_activation = ACTIVATION_FUNCTIONS[self.model.activation]["func"](net_input)
                else: # Output layer
                    # Use the specified output activation function for display
                    current_activation = self._apply_output_activation_for_display(net_input, output_activation_name)

                activations_list.append(current_activation.flatten()) # Flatten for consistent shape
            return activations_list

        return [X.flatten(), np.random.rand(self.output_nodes).flatten() * 0.01] # Fallback


    def partial_fit(self, input_data: List[float], target: List[float], classes: Optional[List[int]] = None) -> float:
        """
        Trains the Sklearn Perceptron/MLPClassifier for one iteration.
        """
        X = np.array(input_data).reshape(1, -1)
        y = np.array(target)

        # Reshape y to be 1D for single-output, or 2D for multi-output if necessary for MLPClassifier
        if self.output_nodes == 1:
            y = y.reshape(1,)
        else:
            y = y.reshape(1, -1) # For multi-output classifiers

        if not self.fitted:
            self.scaler.fit(X)
            # Sklearn Perceptron/MLPClassifier needs to know all possible classes upfront for partial_fit
            # This is crucial. If `classes` is not passed correctly, it will fail on the first `partial_fit`.
            if classes is None:
                # Default classes for binary classification (0, 1). Adjust as needed for multi-class.
                if self.output_nodes == 1: # For single output, assume binary classification
                    classes = [0, 1]
                else: # For multi-output, assume one-hot encoded or similar targets, so pass distinct values
                    # This is a simplification; in a real app, you'd need actual class labels
                    classes = list(range(self.output_nodes))
                    # If target is one-hot, convert it to single class label for partial_fit
                    if len(y.shape) > 1 and y.shape[1] > 1:
                         y = np.argmax(y, axis=1) # Convert one-hot to class label for MLPClassifier.partial_fit


            self.model.partial_fit(self.scaler.transform(X), y, classes=classes)
            self.fitted = True
        else:
            if len(y.shape) > 1 and y.shape[1] > 1 and isinstance(self.model, MLPClassifier):
                y = np.argmax(y, axis=1) # Convert one-hot to class label if MLPClassifier and multi-output

            self.model.partial_fit(self.scaler.transform(X), y)

        # Calculate error (MSE for visualization)
        predictions = self.forward_pass(input_data, self.output_activation_name)[-1]
        error = float(np.mean((predictions - np.array(target))**2))
        return error

    def get_model_data(self) -> List[Dict[str, List[List[float]]]]:
        """Returns the current weights and biases."""
        if not self.fitted:
            # If not fitted, return dummy or empty if model hasn't been initialized internally
            return []

        if isinstance(self.model, SklearnPerceptron):
            if hasattr(self.model, 'coef_') and self.model.coef_ is not None and hasattr(self.model, 'intercept_') and self.model.intercept_ is not None:
                return [{"weights": self.model.coef_.tolist(), "biases": self.model.intercept_.tolist()}]
            return []
        elif isinstance(self.model, MLPClassifier):
            if hasattr(self.model, 'coefs_') and self.model.coefs_ is not None and hasattr(self.model, 'intercepts_') and self.model.intercepts_ is not None:
                return [
                    {"weights": coef.tolist(), "biases": intercept.tolist()}
                    for coef, intercept in zip(self.model.coefs_, self.model.intercepts_)
                ]
            return []
        return []