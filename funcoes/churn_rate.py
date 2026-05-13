#ADASYN-wPSO-NN | Churn Prediction
# Função para o treino do modelo de Churn Rate
# Após treinado exporta o modelo
#Imports
# Foi necessário utilizar versão antigas do pacote para ficarem parelhos
# Com o JupyterNotebook atrelado ao projeto
# numpy==2.0.2 pandas==2.2.2 matplotlib==3.10.0 seaborn==0.13.2
import numpy as np
import pandas as pd
# Talvez o Matplot não funcione
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (f1_score, accuracy_score, precision_score,
                             recall_score, roc_auc_score)
from imblearn.over_sampling import ADASYN

np.random.seed(42)

#Constantes
TARGET_COLUMN = 'Churn'

#RWN (Random Weight Network)
class RWN:
    def __init__(self, n_hidden=10, activation='sigmoid', random_state=42):
        self.n_hidden     = max(1, int(n_hidden))
        self.activation   = activation
        self.random_state = random_state
        self.W_input      = None
        self.b_hidden     = None
        self.W_output     = None
        self.classes_     = None

    def _activate(self, X):
        if self.activation == 'sigmoid':
            return 1.0 / (1.0 + np.exp(-np.clip(X, -500, 500)))
        elif self.activation == 'tanh':
            return np.tanh(X)
        elif self.activation == 'relu':
            return np.maximum(0, X)
        return X

    def _hidden_output(self, X):
        return self._activate(X @ self.W_input + self.b_hidden)

    def fit(self, X, y):
        rng       = np.random.RandomState(self.random_state)
        n_input   = X.shape[1]
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        self.W_input  = rng.randn(n_input, self.n_hidden) * 0.5
        self.b_hidden = rng.randn(1, self.n_hidden) * 0.5
        H = self._hidden_output(X)
        T = np.zeros((len(y), n_classes))
        for i, cls in enumerate(self.classes_):
            T[:, i] = (y == cls).astype(float)
        self.W_output = np.linalg.pinv(H) @ T
        return self

    def predict_proba(self, X):
        H      = self._hidden_output(X)
        output = H @ self.W_output
        exp_o  = np.exp(output - output.max(axis=1, keepdims=True))
        return exp_o / exp_o.sum(axis=1, keepdims=True)

    def predict(self, X):
        indices = np.argmax(self.predict_proba(X), axis=1)
        return self.classes_[indices]

#PSO (Particle Swarm Optimization)
class PSO:
    def __init__(self, n_particles=30, n_iterations=50, n_features=10,
                 n_bits=5, w=0.7, c1=1.5, c2=1.5, v_max=0.3, random_state=42):
        self.n_particles  = n_particles
        self.n_iterations = n_iterations
        self.n_features   = n_features
        self.n_bits       = n_bits
        self.w            = w
        self.c1           = c1
        self.c2           = c2
        self.v_max        = v_max
        self.random_state = random_state
        self.dim          = n_features + n_bits

        self.best_position   = None
        self.best_fitness    = -np.inf
        self.fitness_history = []

    def decode_neurons(self, neuron_part):
        binary_flags = (neuron_part >= 0.5).astype(int)
        if not any(binary_flags):
            return 1
        n_hidden = int(''.join(map(str, binary_flags)), 2)
        return max(1, n_hidden)

    def apply_weights(self, X, weights):
        return X * weights

    def fitness_function(self, particle, X_train, y_train, X_val, y_val):
        try:
            weights     = particle[:self.n_features]
            neuron_part = particle[self.n_features:]
            n_hidden    = self.decode_neurons(neuron_part)

            X_tr_w = self.apply_weights(X_train, weights)
            X_va_w = self.apply_weights(X_val,   weights)

            rwn = RWN(n_hidden=n_hidden, activation='sigmoid',
                      random_state=self.random_state)
            rwn.fit(X_tr_w, y_train)

            y_pred    = rwn.predict(X_va_w)
            f_measure = f1_score(y_val, y_pred, pos_label=1,
                                 average='binary', zero_division=0)
            return f_measure
        except Exception:
            return 0.0

    def optimize(self, X_train, y_train, X_val, y_val, verbose=True):
        rng = np.random.RandomState(self.random_state)
        positions  = rng.uniform(0, 1, (self.n_particles, self.dim))
        velocities = rng.uniform(-self.v_max, self.v_max,
                                 (self.n_particles, self.dim))

        pbest_positions = positions.copy()
        pbest_fitness   = np.full(self.n_particles, -np.inf)
        gbest_position  = positions[0].copy()
        gbest_fitness   = -np.inf

        self.fitness_history = []

        if verbose:
            print("  Iniciando PSO...")
            print(f"  Particulas: {self.n_particles} | "
                  f"Iteracoes: {self.n_iterations} | "
                  f"Dim: {self.dim}")
            print(f"  {'Iter':>5} | {'gbest F1':>10} | "
                  f"{'Media F1':>10} | {'Neuronios':>10}")
            print("  " + "-"*45)

        for it in range(self.n_iterations):
            iter_fitness = []

            for i in range(self.n_particles):
                fit = self.fitness_function(
                    positions[i], X_train, y_train, X_val, y_val)
                iter_fitness.append(fit)

                if fit > pbest_fitness[i]:
                    pbest_fitness[i]   = fit
                    pbest_positions[i] = positions[i].copy()

                if fit > gbest_fitness:
                    gbest_fitness  = fit
                    gbest_position = positions[i].copy()

            self.fitness_history.append(gbest_fitness)

            if verbose and (it % 5 == 0 or it == self.n_iterations - 1):
                nh = self.decode_neurons(gbest_position[self.n_features:])
                print(f"  {it+1:>5} | {gbest_fitness:>10.6f} | "
                      f"{np.mean(iter_fitness):>10.6f} | {nh:>10}")

            r1 = rng.uniform(0, 1, (self.n_particles, self.dim))
            r2 = rng.uniform(0, 1, (self.n_particles, self.dim))

            velocities = (self.w  * velocities
                          + self.c1 * r1 * (pbest_positions - positions)
                          + self.c2 * r2 * (gbest_position  - positions))
            velocities = np.clip(velocities, -self.v_max, self.v_max)
            positions  = np.clip(positions + velocities, 0, 1)

        best_weights  = gbest_position[:self.n_features]
        best_n_hidden = self.decode_neurons(gbest_position[self.n_features:])
        self.best_position = gbest_position
        self.best_fitness  = gbest_fitness

        if verbose:
            print(f"\n  PSO concluido! Melhor F1: {gbest_fitness:.6f} | "
                  f"Neuronios: {best_n_hidden}")

        return {
            'best_weights'  : best_weights,
            'best_n_hidden' : best_n_hidden,
            'best_fitness'  : gbest_fitness,
            'history'       : self.fitness_history
        }

class ADASYN_wPSO_NN:
    def __init__(self,
                 n_particles=30, n_iterations=50, n_bits=5,
                 w=0.7, c1=1.5, c2=1.5, v_max=0.3,
                 n_folds=5, adasyn_neighbors=5,
                 random_state=42, verbose=True):

        self.n_particles      = n_particles
        self.n_iterations     = n_iterations
        self.n_bits           = n_bits
        self.w                = w
        self.c1               = c1
        self.c2               = c2
        self.v_max            = v_max
        self.n_folds          = n_folds
        self.adasyn_neighbors = adasyn_neighbors
        self.random_state     = random_state
        self.verbose          = verbose

        self.fold_results  = []
        self.best_weights  = None
        self.best_n_hidden = None
        self.final_model   = None
        self.scaler        = None

        # NOTE: Melhor acurácia adicionada
        self.melhor_acc    = None

    def run(self, X, y, feature_names=None):
        n_features = X.shape[1]
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True,
                              random_state=self.random_state)
        self.fold_results = []
        all_histories     = []

        print("\n" + "="*65)
        print("        ADASYN-wPSO-NN  |  Predicao de Churn")
        print("="*65)

        best_f1_global = -np.inf

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
            print(f"\nFOLD {fold}/{self.n_folds}")
            print("-"*50)

            X_train_raw = X[train_idx]
            X_test_raw  = X[test_idx]
            y_train     = y[train_idx]
            y_test      = y[test_idx]

            scaler         = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_raw)
            X_test_scaled  = scaler.transform(X_test_raw)

            dist_antes = dict(zip(*np.unique(y_train, return_counts=True)))
            print(f"  Classes antes ADASYN : {dist_antes}")

            try:
                adasyn = ADASYN(n_neighbors=self.adasyn_neighbors,
                                random_state=self.random_state)
                X_res, y_res = adasyn.fit_resample(X_train_scaled, y_train)
            except Exception as e:
                print(f"  ADASYN falhou ({e}), usando dados originais.")
                X_res, y_res = X_train_scaled, y_train

            dist_depois = dict(zip(*np.unique(y_res, return_counts=True)))
            print(f"  Classes apos  ADASYN : {dist_depois}")
            print(f"  Treino: {X_res.shape[0]} amostras | "
                  f"Teste: {X_test_scaled.shape[0]} amostras")

            val_size    = max(1, int(0.2 * len(y_res)))
            X_pso_train = X_res[:-val_size]
            y_pso_train = y_res[:-val_size]
            X_pso_val   = X_res[-val_size:]
            y_pso_val   = y_res[-val_size:]

            pso = PSO(
                n_particles  = self.n_particles,
                n_iterations = self.n_iterations,
                n_features   = n_features,
                n_bits       = self.n_bits,
                w            = self.w,
                c1           = self.c1,
                c2           = self.c2,
                v_max        = self.v_max,
                random_state = self.random_state + fold
            )

            result = pso.optimize(X_pso_train, y_pso_train,
                                  X_pso_val,   y_pso_val,
                                  verbose=self.verbose)

            all_histories.append(result['history'])
            best_weights  = result['best_weights']
            best_n_hidden = result['best_n_hidden']

            X_train_w = X_res        * best_weights
            X_test_w  = X_test_scaled * best_weights

            rwn = RWN(n_hidden=best_n_hidden, activation='sigmoid',
                      random_state=self.random_state)
            rwn.fit(X_train_w, y_res)

            y_pred  = rwn.predict(X_test_w)
            y_proba = rwn.predict_proba(X_test_w)[:, 1]

            acc  = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
            rec  = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
            f1   = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
            auc  = (roc_auc_score(y_test, y_proba)
                    if len(np.unique(y_test)) > 1 else 0.0)

            metrics = {
                'Fold'       : fold,
                'Accuracy'   : acc,
                'Precision'  : prec,
                'Recall'     : rec,
                'F1-Score'   : f1,
                'AUC-ROC'    : auc,
                'N_Hidden'   : best_n_hidden,
                'PSO_Fitness': result['best_fitness']
            }
            self.fold_results.append(metrics)

            print(f"\n  Resultados Fold {fold}:")
            print(f"    Accuracy  : {acc:.4f}")
            print(f"    Precision : {prec:.4f}")
            print(f"    Recall    : {rec:.4f}")
            print(f"    F1-Score  : {f1:.4f}")
            print(f"    AUC-ROC   : {auc:.4f}")
            print(f"    Neuronios : {best_n_hidden}")

            if f1 > best_f1_global:
                best_f1_global     = f1
                self.best_weights  = best_weights
                self.best_n_hidden = best_n_hidden
                self.final_model   = rwn
                self.scaler        = scaler

                self.melhor_acc    = acc

        results_df = pd.DataFrame(self.fold_results)
        self._print_summary(results_df, feature_names)
        # Suprime plot com os resultados de treino por agora
        self._plot(results_df, all_histories, feature_names)
        return results_df

    def _print_summary(self, df, feature_names):
        """Imprime resumo final com media e desvio padrao."""
        print("\n" + "="*65)
        print("         RESULTADOS FINAIS (Cross-Validation)")
        print("="*65)
        for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']:
            print(f"  {col:<12}: {df[col].mean():.4f} +/- {df[col].std():.4f}")
        print(f"\n  Melhor N Neuronios: {self.best_n_hidden}")

        if feature_names is not None and self.best_weights is not None:
            print("\n  Top 5 Features (peso PSO):")
            top = np.argsort(self.best_weights)[::-1][:5]
            for rank, idx in enumerate(top, 1):
                print(f"    {rank}. {feature_names[idx]:<28} "
                      f"peso: {self.best_weights[idx]:.4f}")
        print("="*65)

    def _plot(self, df, histories, feature_names):
        """Gera visualizacoes dos resultados."""
        has_fi = (feature_names is not None and self.best_weights is not None)
        ncols  = 4 if has_fi else 3
        fig, axes = plt.subplots(1, ncols, figsize=(6*ncols, 5))
        fig.suptitle("ADASYN-wPSO-NN | Resultados",
                     fontsize=14, fontweight='bold')
        colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0']
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']

        ax = axes[0]
        x, w = np.arange(len(df)), 0.15
        for i, m in enumerate(metrics):
            ax.bar(x + i*w, df[m], w, label=m,
                   color=colors[i], edgecolor='black', linewidth=0.5)
        ax.set_title("Metricas por Fold", fontweight='bold')
        ax.set_xticks(x + w*2)
        ax.set_xticklabels([f"Fold {i+1}" for i in range(len(df))])
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=7)
        ax.grid(axis='y', alpha=0.3)

        ax = axes[1]
        means = [df[m].mean() for m in metrics]
        stds  = [df[m].std()  for m in metrics]
        bars  = ax.bar(metrics, means, color=colors,
                       edgecolor='black', linewidth=0.5)
        ax.errorbar(metrics, means, yerr=stds, fmt='none',
                    color='black', capsize=5, linewidth=2)
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.02,
                    f"{mean:.3f}", ha='center', fontsize=9, fontweight='bold')
        ax.set_title("Media +/- Desvio Padrao", fontweight='bold')
        ax.set_ylim(0, 1.15)
        ax.grid(axis='y', alpha=0.3)

        ax = axes[2]
        for i, h in enumerate(histories):
            ax.plot(h, label=f"Fold {i+1}", linewidth=1.5)
        ax.set_xlabel("Iteracao")
        ax.set_ylabel("Melhor F1 (gbest)")
        ax.set_title("Convergencia do PSO", fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        if has_fi:
            ax = axes[3]
            top_n = min(15, len(feature_names))
            idx   = np.argsort(self.best_weights)[::-1][:top_n]
            names = [feature_names[i] for i in idx]
            vals  = self.best_weights[idx]
            ax.barh(range(top_n), vals[::-1],
                    color='#2196F3', edgecolor='black', linewidth=0.5)
            ax.set_yticks(range(top_n))
            ax.set_yticklabels(names[::-1], fontsize=8)
            ax.set_xlabel("Peso PSO")
            ax.set_title("Importancia das Features (Top 15)", fontweight='bold')
            ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig("resultados_ADASYN_wPSO_NN.png", dpi=150,
                    bbox_inches='tight')
        # Suprime gráfico por agora
        # plt.show()
        print("Graficos salvos em: resultados_ADASYN_wPSO_NN.png")

    def predict(self, X_new):
        """Prediz novos dados com o melhor modelo encontrado."""
        X_sc = self.scaler.transform(X_new)
        X_w  = X_sc * self.best_weights
        return self.final_model.predict(X_w)

def treinar_modelo() -> ADASYN_wPSO_NN:
    """Retorna instância treinada do modelo."""
    # TODO: Receber dataset de treino como parâmetro?
    # Carregamendo dataset (TEMPORÁRIO)
    url = 'https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv'
    df = pd.read_csv(url)
    print(f"Dataset: {df.shape[0]} linhas x {df.shape[1]} colunas")

    df_proc = df.copy()

    df_proc['TotalCharges'] = pd.to_numeric(df_proc['TotalCharges'], errors='coerce')
    df_proc['TotalCharges'].fillna(df_proc['TotalCharges'].median(), inplace=True)

    df_proc.drop(columns=['customerID'], inplace=True, errors='ignore')

    le = LabelEncoder()
    for col in df_proc.columns:
        if df_proc[col].dtype == 'object' or str(df_proc[col].dtype) == 'bool':
            df_proc[col] = le.fit_transform(df_proc[col].astype(str))

    X = df_proc.drop(columns=[TARGET_COLUMN]).values.astype(float)
    y = df_proc[TARGET_COLUMN].values.astype(int)
    FEATURE_NAMES = df_proc.drop(columns=[TARGET_COLUMN]).columns.tolist()

    print(f"Features: {X.shape[1]} | Amostras: {X.shape[0]}")
    print(f"Distribuicao: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"Dataset carregado: {df.shape[0]} linhas x {df.shape[1]} colunas")
    print(df.head())

    #Preprocessamento
    print(f"\nDistribuicao da variavel alvo ({TARGET_COLUMN}):")
    print(df[TARGET_COLUMN].value_counts())

    df_proc = df.copy()
    le = LabelEncoder()
    for col in df_proc.columns:
        if df_proc[col].dtype == 'object' or str(df_proc[col].dtype) == 'bool':
            df_proc[col] = le.fit_transform(df_proc[col].astype(str))

    X = df_proc.drop(columns=[TARGET_COLUMN]).values.astype(float)
    y = df_proc[TARGET_COLUMN].values.astype(int)
    FEATURE_NAMES = df_proc.drop(columns=[TARGET_COLUMN]).columns.tolist()

    print(f"\nFeatures : {X.shape[1]}")
    print(f"Amostras : {X.shape[0]}")
    print(f"Classes  : {np.unique(y)}")

    #Configuração de hiperparâmetros
    CONFIG = {
        # PSO
        'n_particles'      : 30,   # numero de partículas
        'n_iterations'     : 50,   # máximo de iterações
        'n_bits'           : 5,    # bits para neuronios (2^5 = até 31 neuronios)
        'w'                : 0.7,  # fator de inércia
        'c1'               : 1.5,  # coeficiente cognitivo
        'c2'               : 1.5,  # coeficiente social
        'v_max'            : 0.3,  # velocidade máxima

        # Cross-Validation
        'n_folds'          : 5,

        # ADASYN
        'adasyn_neighbors' : 5,

        # Geral
        'random_state'     : 42,
        'verbose'          : True
    }

    print("\nConfiguracoes do modelo:")
    for k, v in CONFIG.items():
        print(f"  {k:<20}: {v}")

    #Execução do modelo
    model = ADASYN_wPSO_NN(**CONFIG)

    results_df = model.run(
        X             = X,
        y             = y,
        feature_names = FEATURE_NAMES
    )

    #Resultado e exportação
    print("\nTabela de resultados por fold:")
    print(results_df.to_string(index=False))

    print("\nResumo estatistico:")
    print(results_df.describe().round(4).to_string())

    #Importância das features
    if model.best_weights is not None:
        feat_imp = pd.DataFrame({
            'Feature'  : FEATURE_NAMES,
            'Peso_PSO' : model.best_weights
        }).sort_values('Peso_PSO', ascending=False).reset_index(drop=True)

        print("\nRanking de Importancia das Features:")
        print(feat_imp.to_string(index=False))

        feat_imp.to_csv('feature_importance_PSO.csv', index=False)

    results_df.to_csv('resultados_ADASYN_wPSO_NN.csv', index=False)
    print("\nArquivos salvos:")
    print("  - resultados_ADASYN_wPSO_NN.csv")
    print("  - feature_importance_PSO.csv")
    # O RAG não é agêntico, então ele não poderia "ver" essa imagem
    print("  - resultados_ADASYN_wPSO_NN.png")

    # Retorno do modelo
    return model