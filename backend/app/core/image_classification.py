import numpy as np
import pandas as pd
import os
import random
import shutil
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA


class DataManager:
    """
    Classe pour gérer les opérations liées aux données: chargement, 
    échantillonnage, prétraitement et visualisation.
    """
    
    def __init__(self, img_width=150, img_height=150, channels=3):
        """
        Initialise le gestionnaire de données avec les dimensions d'images souhaitées.
        
        Args:
            img_width (int): Largeur des images après redimensionnement
            img_height (int): Hauteur des images après redimensionnement
            channels (int): Nombre de canaux de couleur (3 pour RGB)
        """
        self.img_width = img_width
        self.img_height = img_height
        self.channels = channels
        self.flat_data = []
        self.images = []
        self.targets = []
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def sample_dataset(self, source_dir, target_dir, num_samples=1000):
        """
        Échantillonne aléatoirement un nombre spécifié d'images d'un répertoire source.
        
        Args:
            source_dir (str): Chemin vers le répertoire source contenant toutes les images
            target_dir (str): Chemin vers le répertoire cible où les images échantillonnées seront copiées
            num_samples (int): Nombre d'images à échantillonner
        
        Returns:
            int: Nombre d'images copiées
        """
        os.makedirs(target_dir, exist_ok=True)
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        all_files = []
        
        for file in os.listdir(source_dir):
            file_lower = file.lower()
            if any(file_lower.endswith(ext) for ext in image_extensions):
                all_files.append(file)
        
        total_images = len(all_files)
        if total_images < num_samples:
            print(f"Attention: Seulement {total_images} images trouvées dans {source_dir}, ce qui est inférieur aux {num_samples} demandées")
            selected_files = all_files
        else:
            selected_files = random.sample(all_files, num_samples)
        
        for file in selected_files:
            source_path = os.path.join(source_dir, file)
            target_path = os.path.join(target_dir, file)
            shutil.copy2(source_path, target_path)
        
        print(f"Copié {len(selected_files)} images de {source_dir} vers {target_dir}")
        return len(selected_files)

    def load_images(self, data_dir, categories):
        """
        Charge les images depuis les répertoires spécifiés et les prétraite.
        
        Args:
            data_dir (str): Répertoire de base contenant les sous-répertoires de catégories
            categories (list): Liste des noms de catégories (sous-répertoires)
        """
        self.flat_data = []
        self.images = []
        self.targets = []
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        
        for category in categories:
            class_index = categories.index(category)
            path = os.path.join(data_dir, category)
            
            if not os.path.exists(path):
                print(f"Le chemin {path} n'existe pas. Vérifiez le nom de catégorie.")
                continue
                
            print(f"Chargement des images de la catégorie: {category}")
            for img_name in os.listdir(path):
                try:
                    img_path = os.path.join(path, img_name)
                    
                    # Skip if not a file or not an image
                    if not os.path.isfile(img_path) or not any(img_name.lower().endswith(ext) for ext in image_extensions):
                        continue
                        
                    img_matrix = imread(img_path)
                    # Vérifier si l'image a le bon nombre de dimensions
                    if len(img_matrix.shape) != 3 and self.channels == 3:
                        # Convertir l'image en niveaux de gris en RGB
                        img_matrix = np.stack((img_matrix,)*3, axis=-1)
                    # Redimensionner l'image
                    img_resized = resize(img_matrix, (self.img_height, self.img_width, self.channels))
                    self.flat_data.append(img_resized.flatten())
                    self.images.append(img_resized)
                    # Ensure each image is labeled with its category name
                    self.targets.append(category)
                except Exception as e:
                    print(f"Erreur lors du chargement de l'image {img_name}: {e}")
        
        print(f"Chargement terminé. {len(self.flat_data)} images chargées au total.")
        
        # Verify we have targets for each image
        if len(self.flat_data) == 0:
            print("Aucune image n'a été chargée!")
        elif len(self.targets) != len(self.flat_data):
            print(f"ATTENTION: Le nombre de cibles ({len(self.targets)}) ne correspond pas au nombre d'images ({len(self.flat_data)})!")
        else:
            print(f"Cibles chargées: {len(set(self.targets))} classes uniques")

    def prepare_data(self, test_size=0.25, random_state=42):
        """
        Prépare les données pour l'entraînement en créant un DataFrame et en divisant en ensembles de test et d'entraînement.
        
        Args:
            test_size (float): Proportion des données à utiliser pour le test
            random_state (int): Graine aléatoire pour la reproductibilité
        """
        if len(self.flat_data) == 0:
            print("Aucune donnée à préparer. Chargez d'abord des images avec load_images().")
            return
            
        self.df = pd.DataFrame(self.flat_data)
        self.df['Target'] = self.targets
        
        self.X = self.df.iloc[:, :-1]
        self.y = self.df['Target']
        
        # Ensure we have enough samples for each class
        if len(self.df['Target'].unique()) < 2:
            print(f"ATTENTION: Seulement {len(self.df['Target'].unique())} classe(s) trouvée(s). Au moins 2 classes sont nécessaires pour l'entraînement.")
            return
            
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        
        print(f"Données préparées. Ensemble d'entraînement: {self.X_train.shape}, Ensemble de test: {self.X_test.shape}")
        print(f"Classes dans l'ensemble d'entraînement: {self.y_train.unique()}")

    def visualize_pca(self, n_components=2, save_path=None):
        """
        Visualise les données avec PCA pour réduire la dimensionnalité.
        
        Args:
            n_components (int): Nombre de composantes principales à utiliser
            save_path (str, optional): Chemin pour sauvegarder la figure
        """
        if self.X is None:
            print("Les données ne sont pas encore préparées. Appelez d'abord prepare_data().")
            return
            
        pca = PCA(n_components=n_components)
        x_pca = pca.fit_transform(self.X)
        
        plt.figure(figsize=(10, 8))
        unique_targets = np.unique(self.y)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_targets)))
        
        for i, category in enumerate(unique_targets):
            mask = self.y == category
            plt.scatter(
                x_pca[mask, 0], 
                x_pca[mask, 1], 
                label=category, 
                c=[colors[i]], 
                alpha=0.7
            )
            
        plt.title("Distribution des points des classes après PCA")
        plt.xlabel("Composante principale 1")
        plt.ylabel("Composante principale 2")
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()
        
    def predict_single_image(self, model, image_path):
        """
        Prédit la classe d'une seule image.
        
        Args:
            model: Modèle entraîné à utiliser pour la prédiction
            image_path (str): Chemin vers l'image à prédire
            
        Returns:
            str: Classe prédite pour l'image
        """
        try:
            img = imread(image_path)
            # Redimensionner et aplatir l'image pour la prédiction
            img_resized = resize(img, (self.img_height, self.img_width, self.channels))
            img_flattened = img_resized.flatten().reshape(1, -1)
            
            # Faire la prédiction
            prediction = model.predict(img_flattened)
            return prediction[0]
        except Exception as e:
            print(f"Erreur lors de la prédiction de l'image {image_path}: {e}")
            return None

class ModelManager:
    """
    Classe pour gérer les modèles de machine learning: création, entraînement,
    évaluation et prédiction.
    """
    
    def __init__(self):
        """
        Initialise le gestionnaire de modèles avec un dictionnaire de modèles disponibles.
        """
        self.models = {
            'KNN': {
                'model': KNeighborsClassifier(),
                'params': {
                    'n_neighbors': 5,
                    'weights': 'uniform',
                    'algorithm': 'auto',
                    'leaf_size': 30,
                    'p': 2,  # Distance de Minkowski: 1 pour Manhattan, 2 pour Euclidienne
                    'metric': 'minkowski'
                }
            },
            'SVM': {
                'model': SVC(),
                'params': {
                    'C': 1.0,
                    'kernel': 'rbf',
                    'degree': 3,
                    'gamma': 'scale',
                    'probability': False,
                    'random_state': 42
                }
            },
            'RandomForest': {
                'model': RandomForestClassifier(),
                'params': {
                    'n_estimators': 100,
                    'criterion': 'gini',
                    'max_depth': None,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'max_features': 'sqrt',
                    'random_state': 42
                }
            },
            'DecisionTree': {
                'model': DecisionTreeClassifier(),
                'params': {
                    'criterion': 'gini',
                    'splitter': 'best',
                    'max_depth': None,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'random_state': 42
                }
            },
            'GradientBoosting': {
                'model': GradientBoostingClassifier(),
                'params': {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 3,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'subsample': 1.0,
                    'random_state': 42
                }
            }
        }
        self.current_model = None
        self.model_name = None
        self.trained_models = {}

    def get_available_models(self):
        """
        Retourne les noms des modèles disponibles.
        
        Returns:
            list: Liste des noms de modèles disponibles
        """
        return list(self.models.keys())

    def get_model_params(self, model_name):
        """
        Retourne les paramètres actuels d'un modèle spécifié.
        
        Args:
            model_name (str): Nom du modèle
            
        Returns:
            dict: Dictionnaire des paramètres du modèle
        """
        if model_name in self.models:
            return self.models[model_name]['params'].copy()
        else:
            print(f"Modèle {model_name} non trouvé.")
            return None

    def set_model_params(self, model_name, params):
        """
        Définit de nouveaux paramètres pour un modèle spécifié.
        
        Args:
            model_name (str): Nom du modèle
            params (dict): Nouveaux paramètres à définir
        """
        if model_name in self.models:
            # Mettre à jour seulement les paramètres fournis
            for key, value in params.items():
                if key in self.models[model_name]['params']:
                    self.models[model_name]['params'][key] = value
                else:
                    print(f"Paramètre {key} non reconnu pour le modèle {model_name}.")
        else:
            print(f"Modèle {model_name} non trouvé.")

    def create_model(self, model_name):
        """
        Crée une instance de modèle avec les paramètres actuels.
        
        Args:
            model_name (str): Nom du modèle à créer
            
        Returns:
            object: Instance du modèle créé ou None si le modèle n'est pas disponible
        """
        if model_name in self.models:
            model_class = type(self.models[model_name]['model'])
            params = self.models[model_name]['params']
            self.current_model = model_class(**params)
            self.model_name = model_name
            return self.current_model
        else:
            print(f"Modèle {model_name} non disponible.")
            return None

    def train_model(self, X_train, y_train):
        """
        Entraîne le modèle actuel sur les données fournies.
        
        Args:
            X_train: Caractéristiques d'entraînement
            y_train: Étiquettes d'entraînement
            
        Returns:
            object: Modèle entraîné
        """
        if self.current_model is None:
            print("Aucun modèle n'a été créé. Utilisez create_model() d'abord.")
            return None
            
        print(f"Entraînement du modèle {self.model_name}...")
        self.current_model.fit(X_train, y_train)
        self.trained_models[self.model_name] = self.current_model
        return self.current_model

    def evaluate_model(self, X_test, y_test):
        """
        Évalue le modèle actuel sur les données de test.
        
        Args:
            X_test: Caractéristiques de test
            y_test: Étiquettes de test
            
        Returns:
            dict: Dictionnaire des métriques d'évaluation
        """
        if self.current_model is None:
            print("Aucun modèle n'a été entraîné. Utilisez train_model() d'abord.")
            return None
            
        y_pred = self.current_model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred) * 100
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        
        print(f"{self.model_name} - Précision: {accuracy:.2f}%")
        print(f"{self.model_name} - Matrice de confusion:\n{conf_matrix}")
        print(f"{self.model_name} - Rapport de classification:\n{class_report}")
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'predictions': y_pred
        }

    def predict(self, X):
        """
        Fait des prédictions avec le modèle actuel.
        
        Args:
            X: Données pour lesquelles faire des prédictions
            
        Returns:
            array: Prédictions du modèle
        """
        if self.current_model is None:
            print("Aucun modèle n'a été entraîné. Utilisez train_model() d'abord.")
            return None
            
        return self.current_model.predict(X)

    def get_trained_model(self, model_name):
        """
        Récupère un modèle entraîné par son nom.
        
        Args:
            model_name (str): Nom du modèle entraîné à récupérer
            
        Returns:
            object: Modèle entraîné ou None s'il n'existe pas
        """
        if model_name in self.trained_models:
            return self.trained_models[model_name]
        else:
            print(f"Le modèle {model_name} n'a pas été entraîné.")
            return None


class IndustrialVisionApp:
    """
    Classe principale qui intègre le gestionnaire de données et le gestionnaire de modèles
    pour fournir une interface utilisateur simple.
    """
    
    def __init__(self, img_width=150, img_height=150, channels=3):
        """
        Initialise l'application avec les gestionnaires de données et de modèles.
        
        Args:
            img_width (int): Largeur des images après redimensionnement
            img_height (int): Hauteur des images après redimensionnement
            channels (int): Nombre de canaux de couleur (3 pour RGB)
        """
        self.data_manager = DataManager(img_width, img_height, channels)
        self.model_manager = ModelManager()
        self.last_evaluation = None  # Store the last evaluation results
        
    def sample_dataset(self, source_dir, target_dir, num_samples=1000):
        """
        Échantillonne le dataset pour un équilibre des classes.
        
        Args:
            source_dir (str): Répertoire source
            target_dir (str): Répertoire cible
            num_samples (int): Nombre d'échantillons par classe
        """
        return self.data_manager.sample_dataset(source_dir, target_dir, num_samples)
        
    def load_dataset(self, data_dir, categories):
        """
        Charge le dataset à partir des répertoires spécifiés.
        
        Args:
            data_dir (str): Répertoire de base des données
            categories (list): Liste des catégories (classes)
        """
        self.data_manager.load_images(data_dir, categories)
        self.data_manager.prepare_data()
        
    def visualize_data(self, save_path=None):
        """
        Visualise les données avec PCA.
        
        Args:
            save_path (str, optional): Chemin pour sauvegarder la visualisation
        """
        self.data_manager.visualize_pca(save_path=save_path)
        
    def get_available_models(self):
        """
        Retourne la liste des modèles disponibles.
        
        Returns:
            list: Liste des noms de modèles disponibles
        """
        return self.model_manager.get_available_models()
        
    def get_model_params(self, model_name):
        """
        Retourne les paramètres d'un modèle spécifié.
        
        Args:
            model_name (str): Nom du modèle
            
        Returns:
            dict: Dictionnaire des paramètres
        """
        return self.model_manager.get_model_params(model_name)
        
    def set_model_params(self, model_name, params):
        """
        Définit les paramètres d'un modèle.
        
        Args:
            model_name (str): Nom du modèle
            params (dict): Nouveaux paramètres
        """
        self.model_manager.set_model_params(model_name, params)
        
    def train_model(self, model_name):
        """
        Crée et entraîne un modèle spécifié.
        
        Args:
            model_name (str): Nom du modèle à entraîner
            
        Returns:
            dict: Résultats de l'évaluation du modèle
        """
        model = self.model_manager.create_model(model_name)
        if model is None:
            return None
            
        self.model_manager.train_model(self.data_manager.X_train, self.data_manager.y_train)
        results = self.model_manager.evaluate_model(self.data_manager.X_test, self.data_manager.y_test)
        
        self.last_evaluation = results
        return results
        
    def predict_image(self, model_name, image_path):
        """
        Prédit la classe d'une image avec un modèle spécifié.
        
        Args:
            model_name (str): Nom du modèle à utiliser pour la prédiction
            image_path (str): Chemin vers l'image à prédire
            
        Returns:
            str: Classe prédite pour l'image
        """
        model = self.model_manager.get_trained_model(model_name)
        if model is None:
            return None
            
        return self.data_manager.predict_single_image(model, image_path)
        
    def compare_models(self, model_names=None):
        """
        Compare plusieurs modèles sur le même ensemble de données.
        
        Args:
            model_names (list, optional): Liste des noms de modèles à comparer.
                Si None, tous les modèles disponibles sont comparés.
                
        Returns:
            dict: Dictionnaire des résultats de comparaison
        """
        if model_names is None:
            model_names = self.get_available_models()
            
        results = {}
        
        for model_name in model_names:
            print(f"\nÉvaluation du modèle {model_name}...")
            model_results = self.train_model(model_name)
            if model_results:
                results[model_name] = model_results
                
        # Comparaison des précisions
        accuracies = {name: res['accuracy'] for name, res in results.items()}
        best_model = max(accuracies, key=accuracies.get)
        
        print("\n=== Comparaison des modèles ===")
        for name, acc in accuracies.items():
            print(f"{name}: {acc:.2f}%")
        print(f"\nMeilleur modèle: {best_model} avec {accuracies[best_model]:.2f}% de précision")
        
        return results


# Exemple d'utilisation:
if __name__ == "__main__":
    # Création de l'application
    app = IndustrialVisionApp(img_width=150, img_height=150)
    
    # Échantillonnage du dataset (si nécessaire)
    base_dir = "Industrial-Equipment"
    app.sample_dataset(
        os.path.join(base_dir, "Defected"),
        os.path.join(base_dir, "Defected_sampled"),
        1000
    )
    app.sample_dataset(
        os.path.join(base_dir, "Non-Defected"),
        os.path.join(base_dir, "Non-Defected_sampled"),
        1000
    )
    
    # Chargement du dataset
    app.load_dataset(
        base_dir, 
        ["Defected_sampled", "Non-Defected_sampled"]
    )
    
    # Visualisation des données
    app.visualize_data(save_path="class_distribution_pca.png")
    
    # Obtention de la liste des modèles disponibles
    models = app.get_available_models()
    print(f"Modèles disponibles: {models}")
    
    # Personnalisation des paramètres d'un modèle
    # app.set_model_params("KNN", {"n_neighbors": 124, "metric": "euclidean"})
    
    # Entraînement et évaluation d'un modèle spécifique
    results_gb = app.train_model("GradientBoosting")
    
    # Entraînement et comparaison de plusieurs modèles
    comparison_results = app.compare_models(["RandomForest", "GradientBoosting"])
    
    # Prédiction sur une nouvelle image
    defect_image_path = 'i-defective_machines_tools (1).jpg'
    prediction = app.predict_image("GradientBoosting", defect_image_path)
    print(f"Prédiction KNN pour l'image défectueuse: {prediction}")
    
    non_defect_image_path = 'Non-defective_machine.jpg'
    prediction = app.predict_image("GradientBoosting", non_defect_image_path)
    print(f"Prédiction RandomForest pour l'image non défectueuse: {prediction}")
