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
from typing import List, Literal, Dict, Any
import random

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
    
    
    
class LayerModel(BaseModel):
    weights: List[List[float]]
    biases: List[float]

class NetworkConfig(BaseModel):
    inputSize: int
    hiddenSize: int # For hidden layers
    outputNodes: int
    hiddenLayers: int # Number of hidden layers
    activation: Literal["sigmoid", "relu", "identity","softmax"] # Activation for hidden layers
    outputActivation: Literal["sigmoid", "relu", "identity", "softmax"] # NEW: Activation for output layer
    learningRate: float

class TrainRequest(BaseModel):
    input: List[List[float]]
    target: List[List[float]]
    network: List[LayerModel]
    config: NetworkConfig

class TrainResponse(BaseModel):
    network: List[LayerModel]
    error: float
    accuracy: float
    deltas: List[List[float]]
    full_activations: List[List[float]] # Store all z and a values (flattened to list of lists)

class CustomPerceptron:
    def __init__(self, input_size: Optional[int] = None, hidden_size: Optional[int] = None,
                 output_nodes: Optional[int] = None, hidden_layers: int = 0,
                 activation_name: str = "sigmoid", output_activation_name: str = "sigmoid", # NEW PARAMETER
                 layers_data: Optional[List[LayerModel]] = None):
        self.hidden_activation_name = activation_name
        self.output_activation_name = output_activation_name
        self.activations = {
            "sigmoid": self._sigmoid,
            "relu": self._relu,
            "identity": self._identity,
            "softmax": self._softmax
        }
        self.derivatives = {
            "sigmoid": self._sigmoid_derivative,
            "relu": self._relu_derivative,
            "identity": self._identity_derivative,
            # Note: For Softmax with MSE, the derivative is complex (Jacobian).
            # This simplified derivative is for chain rule if treated element-wise, but problematic for true Softmax+MSE.
            # Cross-entropy is preferred with Softmax.
            "softmax": self._softmax_derivative
        }

        if layers_data:
            self.layers = []
            for layer_model in layers_data:
                self.layers.append({
                    "weights": np.array(layer_model.weights),
                    "biases": np.array(layer_model.biases)
                })
        else:
            self.layers = []
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

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(x, -500, 500))) # Add clipping for numerical stability

    def _sigmoid_derivative(self, y: np.ndarray) -> np.ndarray:
        return y * (1 - y)

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def _relu_derivative(self, y: np.ndarray) -> np.ndarray:
        return (y > 0).astype(float)

    def _identity(self, x: np.ndarray) -> np.ndarray:
        return x

    def _identity_derivative(self, y: np.ndarray) -> np.ndarray:
        return np.ones_like(y)
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def _softmax_derivative(self, activated_output: np.ndarray) -> np.ndarray:
        # This derivative is appropriate when softmax is followed by Cross-Entropy Loss
        # and you want (y_hat - y). If used with MSE, it's problematic.
        # For MSE, the d/dz_i (softmax_i) is s_i(1-s_i) + sum_{j!=i} (-s_i*s_j). This is not just s_i(1-s_i).
        # To avoid overcomplicating, for MSE, we are essentially treating it element-wise like sigmoid.
        # This will be `activated_output * (1 - activated_output)` for individual elements for simplified chain rule.
        # However, for accurate softmax derivative in general, it's a Jacobian.
        # Given the current structure, using y*(1-y) is the closest simplified element-wise derivative.
        return activated_output * (1 - activated_output) # Simplified for element-wise chain rule

    def get_activation_function(self, layer_idx: int):
        if layer_idx == len(self.layers) - 1: # Last layer
            return self.activations.get(self.output_activation_name, self._sigmoid)
        return self.activations.get(self.hidden_activation_name, self._sigmoid)

    def get_activation_derivative(self, layer_idx: int, activated_output: np.ndarray):
        if layer_idx == len(self.layers) - 1: # Last layer
            act_name = self.output_activation_name
        else:
            act_name = self.hidden_activation_name
        
        # Pass the activated output to the derivative function
        if act_name == "sigmoid":
            return self._sigmoid_derivative(activated_output)
        elif act_name == "relu":
            return self._relu_derivative(activated_output)
        elif act_name == "identity":
            return self._identity_derivative(activated_output)
        elif act_name == "softmax":
            return self._softmax_derivative(activated_output) # Use the (simplified) softmax derivative

    def forward_pass(self, input_data: List[float]):
        activations_list = [] # Will store a_0, z_1, a_1, z_2, a_2, ..., z_L, a_L
        current_activation = np.array(input_data)
        activations_list.append(current_activation) # a_0 (input layer's output)

        for i, layer_data in enumerate(self.layers):
            weights = layer_data["weights"]
            biases = layer_data["biases"]

            z = np.dot(weights, current_activation) + biases
            activations_list.append(z) # z_{i+1} (raw output of current layer)

            current_activation = self.get_activation_function(i)(z) # a_{i+1} (activated output of current layer)
            activations_list.append(current_activation)

        return activations_list

    def backward_pass(self, input_data, target, activations_list_raw, learning_rate):
        deltas = [None] * len(self.layers) # deltas[i] corresponds to the delta for self.layers[i]

        # Output layer delta calculation (last layer in self.layers, index L-1)
        # a_L is activations_list_raw[-1], z_L is activations_list_raw[-2]
        output_activations = activations_list_raw[-1] # a_L
        target = np.array(target).reshape(output_activations.shape)

        loss_derivative = 2 * (output_activations - target) # dC/da_L (for MSE)

        # da_L/dz_L using the output layer's specific activation derivative
        output_layer_activation_derivative = self.get_activation_derivative(len(self.layers) - 1, output_activations)
        deltas[len(self.layers) - 1] = loss_derivative * output_layer_activation_derivative

        # Backpropagate through hidden layers (from second-to-last layer down to the first hidden layer)
        # Iterate from the (L-2)th layer down to the 0th layer (first hidden layer)
        for i in reversed(range(len(self.layers) - 1)):
            current_layer_idx = i # This `i` is the 0-indexed layer in `self.layers`

            next_layer_weights = self.layers[current_layer_idx + 1]['weights'] # W_{i+1}
            
            # error_prop is (W_{i+1})^T * delta_{i+1}
            error_prop = np.dot(next_layer_weights.T, deltas[current_layer_idx + 1])

            # activated output of the current layer (a_{i+1})
            # a_k is at index 2*k. Here k = current_layer_idx + 1
            current_layer_activations_a = activations_list_raw[(current_layer_idx + 1) * 2] 

            # da_{i+1}/dz_{i+1} using the current layer's activation derivative
            current_layer_activation_derivative = self.get_activation_derivative(current_layer_idx, current_layer_activations_a)
            
            deltas[current_layer_idx] = error_prop * current_layer_activation_derivative

        # Update weights and biases for all layers
        new_layers_data = []
        for i in range(len(self.layers)):
            current_layer_data = self.layers[i]
            
            if i == 0: # For the first layer, previous activations are the original input_data (a_0)
                prev_activations = np.array(input_data)
            else: # For subsequent layers, previous activations are the activated outputs of the (i-1)th layer (a_i)
                # a_i is at index `i*2` in activations_list_raw (this is the output of the *previous* layer that feeds *this* layer)
                prev_activations = activations_list_raw[i * 2]

            delta_reshaped = np.array(deltas[i]).reshape(-1, 1)
            prev_act_reshaped = prev_activations.reshape(1, -1)

            weight_update = np.dot(delta_reshaped, prev_act_reshaped)
            
            new_weights = current_layer_data['weights'] - learning_rate * weight_update
            new_biases = current_layer_data['biases'] - learning_rate * deltas[i]
            
            new_layers_data.append({
                "weights": new_weights.tolist(),
                "biases": new_biases.tolist()
            })

        self.layers = new_layers_data # Update the instance's layers with new weights/biases
        return [d.tolist() for d in deltas if d is not None]

# End of Custom Perceptron specific code

# Initialize FastAPI app and CORS (from original main.py)
# ...

@app.post("/initialize")
async def initialize_network(config: NetworkConfig):
    """Initializes a new CustomPerceptron network with specified configuration."""
    try:
        perceptron = CustomPerceptron(
            input_size=config.inputSize,
            hidden_size=config.hiddenSize,
            output_nodes=config.outputNodes,
            hidden_layers=config.hiddenLayers,
            activation_name=config.activation,
            output_activation_name=config.outputActivation # Pass new param
        )
        return [LayerModel(weights=layer["weights"].tolist(), biases=layer["biases"].tolist()) for layer in perceptron.layers]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/train")
async def train_network(req: TrainRequest):
    """
    Trains the CustomPerceptron network for one iteration.
    Receives the current network state and training configuration from the frontend.
    """
    perceptron = CustomPerceptron(
        activation_name=req.config.activation,
        output_activation_name=req.config.outputActivation, # Pass new param
        layers_data=req.network
    )
    
    activations_list = perceptron.forward_pass(req.input[0])
    
    # Backward pass and update
    deltas = perceptron.backward_pass(
        input_data=req.input[0],
        target=req.target[0],
        activations_list_raw=activations_list,
        learning_rate=req.config.learningRate
    )
    
    error = float(np.mean((activations_list[-1] - req.target[0])**2))

    return TrainResponse(
        network=[LayerModel(weights=layer["weights"].tolist(), biases=layer["biases"].tolist()) for layer in perceptron.layers],
        error=error,
        accuracy=0.0, # Accuracy calculation would need to be added if desired
        deltas=deltas,
        full_activations=[act.tolist() for act in activations_list] # Ensure all elements are lists
    )