"""
Module de prétraitement des données Home Credit.
=================================================

Ce module contient le pipeline complet de prétraitement:
- Chargement et agrégation des tables
- Feature engineering
- Encodage des variables catégorielles
- Gestion des valeurs manquantes
- Normalisation
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
import warnings

warnings.filterwarnings('ignore')

# Chemins
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"


def clean_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie les noms de colonnes en remplaçant les caractères spéciaux.
    
    Args:
        df: DataFrame avec colonnes à nettoyer
        
    Returns:
        DataFrame avec noms de colonnes nettoyés
    """
    df = df.copy()
    df.columns = df.columns.str.replace('[', '_', regex=False)
    df.columns = df.columns.str.replace(']', '_', regex=False)
    df.columns = df.columns.str.replace('(', '_', regex=False)
    df.columns = df.columns.str.replace(')', '_', regex=False)
    df.columns = df.columns.str.replace('<', '_', regex=False)
    df.columns = df.columns.str.replace('>', '_', regex=False)
    df.columns = df.columns.str.replace(',', '_', regex=False)
    df.columns = df.columns.str.replace(':', '_', regex=False)
    df.columns = df.columns.str.replace('/', '_', regex=False)
    df.columns = df.columns.str.replace(' ', '_', regex=False)
    return df


def reduce_memory_usage(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Réduit l'utilisation mémoire d'un DataFrame en optimisant les types de données.
    
    Args:
        df: DataFrame à optimiser
        verbose: Afficher les informations de réduction
        
    Returns:
        DataFrame optimisé
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    
    if verbose:
        end_mem = df.memory_usage(deep=True).sum() / 1024**2
        print(f'Memory usage decreased to {end_mem:5.2f} MB ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
    
    return df


def load_application_data(data_dir: Path = DATA_DIR) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Charge les données application_train et application_test.
    
    Returns:
        Tuple (train_df, test_df)
    """
    train_path = data_dir / "application_train.csv"
    test_path = data_dir / "application_test.csv"
    
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Données non trouvées dans {data_dir}. Exécutez scripts/download_data.py")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"✅ Données chargées: train={train_df.shape}, test={test_df.shape}")
    return train_df, test_df


def load_bureau_data(data_dir: Path = DATA_DIR) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Charge les données bureau et bureau_balance."""
    bureau = pd.read_csv(data_dir / "bureau.csv")
    bureau_balance = pd.read_csv(data_dir / "bureau_balance.csv")
    return bureau, bureau_balance


def load_previous_application(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Charge les données previous_application."""
    return pd.read_csv(data_dir / "previous_application.csv")


def load_installments(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Charge les données installments_payments."""
    return pd.read_csv(data_dir / "installments_payments.csv")


def load_pos_cash(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Charge les données POS_CASH_balance."""
    return pd.read_csv(data_dir / "POS_CASH_balance.csv")


def load_credit_card(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Charge les données credit_card_balance."""
    return pd.read_csv(data_dir / "credit_card_balance.csv")


def aggregate_bureau_data(bureau: pd.DataFrame, bureau_balance: pd.DataFrame) -> pd.DataFrame:
    """
    Agrège les données bureau au niveau client (SK_ID_CURR).
    
    Crée des features statistiques sur l'historique de crédit.
    """
    # Agrégation bureau_balance par SK_ID_BUREAU
    bb_agg = bureau_balance.groupby('SK_ID_BUREAU').agg({
        'MONTHS_BALANCE': ['min', 'max', 'size'],
        'STATUS': lambda x: (x == 'C').sum()  # Nombre de mois "closed"
    })
    bb_agg.columns = ['BB_MONTHS_MIN', 'BB_MONTHS_MAX', 'BB_COUNT', 'BB_STATUS_CLOSED']
    bb_agg = bb_agg.reset_index()
    
    # Joindre avec bureau
    bureau = bureau.merge(bb_agg, on='SK_ID_BUREAU', how='left')
    
    # Agrégation bureau par SK_ID_CURR
    num_cols = bureau.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in ['SK_ID_CURR', 'SK_ID_BUREAU']]
    
    agg_dict = {col: ['mean', 'max', 'min', 'sum'] for col in num_cols[:10]}  # Limiter pour performance
    agg_dict['SK_ID_BUREAU'] = 'count'
    
    bureau_agg = bureau.groupby('SK_ID_CURR').agg(agg_dict)
    bureau_agg.columns = ['BUREAU_' + '_'.join(col).upper() for col in bureau_agg.columns]
    bureau_agg = bureau_agg.reset_index()
    
    return bureau_agg


def aggregate_previous_application(prev_app: pd.DataFrame) -> pd.DataFrame:
    """
    Agrège les données previous_application au niveau client.
    """
    # Statistiques sur les applications précédentes
    num_cols = prev_app.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in ['SK_ID_CURR', 'SK_ID_PREV']]
    
    agg_dict = {col: ['mean', 'max', 'min'] for col in num_cols[:8]}
    agg_dict['SK_ID_PREV'] = 'count'
    
    # Comptage par statut
    prev_app['PREV_APPROVED'] = (prev_app['NAME_CONTRACT_STATUS'] == 'Approved').astype(int)
    prev_app['PREV_REFUSED'] = (prev_app['NAME_CONTRACT_STATUS'] == 'Refused').astype(int)
    
    agg_dict['PREV_APPROVED'] = 'sum'
    agg_dict['PREV_REFUSED'] = 'sum'
    
    prev_agg = prev_app.groupby('SK_ID_CURR').agg(agg_dict)
    prev_agg.columns = ['PREV_' + '_'.join(col).upper() if isinstance(col, tuple) else 'PREV_' + col 
                        for col in prev_agg.columns]
    prev_agg = prev_agg.reset_index()
    
    return prev_agg


def aggregate_installments(installments: pd.DataFrame) -> pd.DataFrame:
    """
    Agrège les données installments_payments au niveau client.
    """
    # Calculer les retards de paiement
    installments['PAYMENT_DIFF'] = installments['AMT_PAYMENT'] - installments['AMT_INSTALMENT']
    installments['DAYS_LATE'] = installments['DAYS_ENTRY_PAYMENT'] - installments['DAYS_INSTALMENT']
    installments['LATE_PAYMENT'] = (installments['DAYS_LATE'] > 0).astype(int)
    
    agg_dict = {
        'NUM_INSTALMENT_VERSION': ['mean', 'max'],
        'NUM_INSTALMENT_NUMBER': ['max'],
        'DAYS_LATE': ['mean', 'max', 'sum'],
        'PAYMENT_DIFF': ['mean', 'sum'],
        'LATE_PAYMENT': ['sum', 'mean'],
        'AMT_PAYMENT': ['mean', 'sum']
    }
    
    inst_agg = installments.groupby('SK_ID_CURR').agg(agg_dict)
    inst_agg.columns = ['INST_' + '_'.join(col).upper() for col in inst_agg.columns]
    inst_agg = inst_agg.reset_index()
    
    return inst_agg


def aggregate_pos_cash(pos_cash: pd.DataFrame) -> pd.DataFrame:
    """
    Agrège les données POS_CASH_balance au niveau client.
    """
    pos_cash['POS_LATE'] = (pos_cash['SK_DPD'] > 0).astype(int)
    
    agg_dict = {
        'MONTHS_BALANCE': ['mean', 'max'],
        'SK_DPD': ['mean', 'max', 'sum'],
        'SK_DPD_DEF': ['mean', 'max', 'sum'],
        'POS_LATE': ['sum', 'mean'],
        'CNT_INSTALMENT': ['mean', 'max'],
        'CNT_INSTALMENT_FUTURE': ['mean', 'min']
    }
    
    pos_agg = pos_cash.groupby('SK_ID_CURR').agg(agg_dict)
    pos_agg.columns = ['POS_' + '_'.join(col).upper() for col in pos_agg.columns]
    pos_agg = pos_agg.reset_index()
    
    return pos_agg


def aggregate_credit_card(credit_card: pd.DataFrame) -> pd.DataFrame:
    """
    Agrège les données credit_card_balance au niveau client.
    """
    credit_card['CC_UTILIZATION'] = credit_card['AMT_BALANCE'] / (credit_card['AMT_CREDIT_LIMIT_ACTUAL'] + 1)
    credit_card['CC_LATE'] = (credit_card['SK_DPD'] > 0).astype(int)
    
    agg_dict = {
        'MONTHS_BALANCE': ['mean', 'max'],
        'AMT_BALANCE': ['mean', 'max'],
        'AMT_CREDIT_LIMIT_ACTUAL': ['mean', 'max'],
        'AMT_DRAWINGS_CURRENT': ['mean', 'sum'],
        'AMT_PAYMENT_CURRENT': ['mean', 'sum'],
        'CC_UTILIZATION': ['mean', 'max'],
        'CC_LATE': ['sum', 'mean'],
        'SK_DPD': ['mean', 'max']
    }
    
    cc_agg = credit_card.groupby('SK_ID_CURR').agg(agg_dict)
    cc_agg.columns = ['CC_' + '_'.join(col).upper() for col in cc_agg.columns]
    cc_agg = cc_agg.reset_index()
    
    return cc_agg


def create_application_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée des features à partir des données application.
    
    Features créées:
    - Ratios financiers
    - Conversion de jours en années
    - Indicateurs de risque
    """
    df = df.copy()
    
    # Ratios financiers
    df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / (df['AMT_INCOME_TOTAL'] + 1)
    df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + 1)
    df['CREDIT_GOODS_RATIO'] = df['AMT_CREDIT'] / (df['AMT_GOODS_PRICE'] + 1)
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / (df['CNT_FAM_MEMBERS'] + 1)
    df['ANNUITY_LENGTH'] = df['AMT_CREDIT'] / (df['AMT_ANNUITY'] + 1)
    
    # Conversion jours en années (valeurs négatives = jours avant application)
    df['AGE_YEARS'] = -df['DAYS_BIRTH'] / 365
    df['EMPLOYED_YEARS'] = -df['DAYS_EMPLOYED'] / 365
    df['EMPLOYED_YEARS'] = df['EMPLOYED_YEARS'].replace({365243 / 365: np.nan})  # Valeur anomalie
    
    # Ratio emploi/âge
    df['EMPLOYED_TO_AGE_RATIO'] = df['EMPLOYED_YEARS'] / (df['AGE_YEARS'] + 1)
    
    # Score externe moyen
    ext_cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    df['EXT_SOURCE_MEAN'] = df[ext_cols].mean(axis=1)
    df['EXT_SOURCE_STD'] = df[ext_cols].std(axis=1)
    df['EXT_SOURCE_MIN'] = df[ext_cols].min(axis=1)
    df['EXT_SOURCE_MAX'] = df[ext_cols].max(axis=1)
    
    # Indicateurs de documents fournis
    doc_cols = [c for c in df.columns if c.startswith('FLAG_DOCUMENT')]
    df['DOCUMENTS_COUNT'] = df[doc_cols].sum(axis=1)
    
    # Indicateurs de contact
    contact_cols = ['FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 
                   'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL']
    contact_cols = [c for c in contact_cols if c in df.columns]
    df['CONTACTS_COUNT'] = df[contact_cols].sum(axis=1)
    
    return df


def encode_categorical_features(
    df: pd.DataFrame, 
    label_encoders: Optional[Dict[str, LabelEncoder]] = None,
    fit: bool = True
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Encode les variables catégorielles avec LabelEncoder.
    
    Args:
        df: DataFrame à encoder
        label_encoders: Dictionnaire d'encodeurs existants (pour transform)
        fit: Si True, fit les encodeurs. Si False, utilise les existants.
        
    Returns:
        Tuple (DataFrame encodé, dictionnaire des encodeurs)
    """
    df = df.copy()
    
    if label_encoders is None:
        label_encoders = {}
    
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    for col in cat_cols:
        if fit:
            le = LabelEncoder()
            # Ajouter une catégorie pour les valeurs inconnues
            df[col] = df[col].fillna('MISSING')
            le.fit(df[col].astype(str))
            label_encoders[col] = le
        else:
            le = label_encoders.get(col)
            if le is None:
                continue
            df[col] = df[col].fillna('MISSING')
            # Gérer les catégories inconnues
            known_classes = set(le.classes_)
            df[col] = df[col].astype(str).apply(
                lambda x: x if x in known_classes else 'MISSING'
            )
        
        df[col] = le.transform(df[col].astype(str))
    
    return df, label_encoders


def build_full_dataset(
    data_dir: Path = DATA_DIR,
    include_supplementary: bool = True,
    sample_frac: Optional[float] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construit le dataset complet avec toutes les features agrégées.
    
    Args:
        data_dir: Chemin vers les données
        include_supplementary: Si True, inclut bureau, previous_app, etc.
        sample_frac: Fraction des données à utiliser (pour tests rapides)
        
    Returns:
        Tuple (train_df, test_df) avec toutes les features
    """
    print("📊 Construction du dataset complet...")
    
    # Charger les données principales
    train_df, test_df = load_application_data(data_dir)
    
    if sample_frac:
        train_df = train_df.sample(frac=sample_frac, random_state=42)
        test_df = test_df.sample(frac=sample_frac, random_state=42)
        print(f"   ⚡ Échantillonnage: {sample_frac*100}%")
    
    # Feature engineering sur application
    print("   🔧 Feature engineering application...")
    train_df = create_application_features(train_df)
    test_df = create_application_features(test_df)
    
    if include_supplementary:
        # Charger et agréger les tables supplémentaires
        try:
            print("   📁 Agrégation bureau...")
            bureau, bureau_balance = load_bureau_data(data_dir)
            bureau_agg = aggregate_bureau_data(bureau, bureau_balance)
            train_df = train_df.merge(bureau_agg, on='SK_ID_CURR', how='left')
            test_df = test_df.merge(bureau_agg, on='SK_ID_CURR', how='left')
            del bureau, bureau_balance, bureau_agg
        except Exception as e:
            print(f"   ⚠️ Bureau non chargé: {e}")
        
        try:
            print("   📁 Agrégation previous_application...")
            prev_app = load_previous_application(data_dir)
            prev_agg = aggregate_previous_application(prev_app)
            train_df = train_df.merge(prev_agg, on='SK_ID_CURR', how='left')
            test_df = test_df.merge(prev_agg, on='SK_ID_CURR', how='left')
            del prev_app, prev_agg
        except Exception as e:
            print(f"   ⚠️ Previous application non chargé: {e}")
        
        try:
            print("   📁 Agrégation installments...")
            installments = load_installments(data_dir)
            inst_agg = aggregate_installments(installments)
            train_df = train_df.merge(inst_agg, on='SK_ID_CURR', how='left')
            test_df = test_df.merge(inst_agg, on='SK_ID_CURR', how='left')
            del installments, inst_agg
        except Exception as e:
            print(f"   ⚠️ Installments non chargé: {e}")
        
        try:
            print("   📁 Agrégation POS_CASH...")
            pos_cash = load_pos_cash(data_dir)
            pos_agg = aggregate_pos_cash(pos_cash)
            train_df = train_df.merge(pos_agg, on='SK_ID_CURR', how='left')
            test_df = test_df.merge(pos_agg, on='SK_ID_CURR', how='left')
            del pos_cash, pos_agg
        except Exception as e:
            print(f"   ⚠️ POS_CASH non chargé: {e}")
        
        try:
            print("   📁 Agrégation credit_card...")
            credit_card = load_credit_card(data_dir)
            cc_agg = aggregate_credit_card(credit_card)
            train_df = train_df.merge(cc_agg, on='SK_ID_CURR', how='left')
            test_df = test_df.merge(cc_agg, on='SK_ID_CURR', how='left')
            del credit_card, cc_agg
        except Exception as e:
            print(f"   ⚠️ Credit card non chargé: {e}")
    
    print(f"✅ Dataset construit: train={train_df.shape}, test={test_df.shape}")
    return train_df, test_df


class CreditScoringPreprocessor:
    """
    Pipeline de prétraitement pour le scoring crédit.
    
    Cette classe encapsule tout le prétraitement:
    - Encodage catégoriel
    - Imputation des valeurs manquantes
    - Normalisation optionnelle
    """
    
    def __init__(
        self,
        impute_strategy: str = 'median',
        scale: bool = False,
        drop_high_nan: float = 0.8
    ):
        """
        Initialise le préprocesseur.
        
        Args:
            impute_strategy: Stratégie d'imputation ('mean', 'median', 'most_frequent')
            scale: Si True, normalise les features numériques
            drop_high_nan: Seuil de NaN pour supprimer une colonne (0.8 = 80%)
        """
        self.impute_strategy = impute_strategy
        self.scale = scale
        self.drop_high_nan = drop_high_nan
        
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.imputer: Optional[SimpleImputer] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = []
        self.columns_to_drop: List[str] = []
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'CreditScoringPreprocessor':
        """
        Fit le préprocesseur sur les données d'entraînement.
        """
        X = X.copy()
        
        # Identifier les colonnes avec trop de NaN
        nan_ratio = X.isnull().sum() / len(X)
        self.columns_to_drop = nan_ratio[nan_ratio > self.drop_high_nan].index.tolist()
        
        # Ajouter SK_ID_CURR aux colonnes à supprimer (pas une feature)
        if 'SK_ID_CURR' in X.columns:
            self.columns_to_drop.append('SK_ID_CURR')
        if 'TARGET' in X.columns:
            self.columns_to_drop.append('TARGET')
        
        X = X.drop(columns=[c for c in self.columns_to_drop if c in X.columns])
        
        # Encoder les catégorielles
        X, self.label_encoders = encode_categorical_features(X, fit=True)
        
        # Fit imputer
        self.imputer = SimpleImputer(strategy=self.impute_strategy)
        self.imputer.fit(X)
        
        # Fit scaler si demandé
        if self.scale:
            X_imputed = self.imputer.transform(X)
            self.scaler = StandardScaler()
            self.scaler.fit(X_imputed)
        
        self.feature_names = X.columns.tolist()
        self.is_fitted = True
        
        print(f"✅ Préprocesseur fitted: {len(self.feature_names)} features")
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transforme les données avec le préprocesseur fitté.
        """
        if not self.is_fitted:
            raise ValueError("Le préprocesseur doit être fitté avant transform()")
        
        X = X.copy()
        
        # Supprimer les colonnes
        X = X.drop(columns=[c for c in self.columns_to_drop if c in X.columns])
        
        # S'assurer que toutes les colonnes sont présentes
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = np.nan
        
        # Garder seulement les features connues dans le bon ordre
        X = X[self.feature_names]
        
        # Encoder les catégorielles
        X, _ = encode_categorical_features(X, self.label_encoders, fit=False)
        
        # Imputer
        X_transformed = self.imputer.transform(X)
        
        # Scaler si configuré
        if self.scale and self.scaler:
            X_transformed = self.scaler.transform(X_transformed)
        
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> np.ndarray:
        """Fit et transform en une seule opération."""
        return self.fit(X, y).transform(X)
    
    def save(self, path: Path):
        """Sauvegarde le préprocesseur."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        print(f"✅ Préprocesseur sauvegardé: {path}")
    
    @classmethod
    def load(cls, path: Path) -> 'CreditScoringPreprocessor':
        """Charge un préprocesseur sauvegardé."""
        preprocessor = joblib.load(path)
        print(f"✅ Préprocesseur chargé: {path}")
        return preprocessor


def prepare_train_test_data(
    data_dir: Path = DATA_DIR,
    include_supplementary: bool = True,
    sample_frac: Optional[float] = None,
    save_preprocessor: bool = True
) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.DataFrame, CreditScoringPreprocessor]:
    """
    Prépare les données pour l'entraînement.
    
    Returns:
        Tuple (X_train, X_test, y_train, test_ids, preprocessor)
    """
    # Construire le dataset
    train_df, test_df = build_full_dataset(
        data_dir=data_dir,
        include_supplementary=include_supplementary,
        sample_frac=sample_frac
    )
    
    # Séparer target et features
    y_train = train_df['TARGET']
    test_ids = test_df[['SK_ID_CURR']]
    
    # Fit et transform
    preprocessor = CreditScoringPreprocessor(
        impute_strategy='median',
        scale=False,
        drop_high_nan=0.8
    )
    
    X_train = preprocessor.fit_transform(train_df)
    X_test = preprocessor.transform(test_df)
    
    # Sauvegarder le préprocesseur
    if save_preprocessor:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        preprocessor.save(MODELS_DIR / "preprocessor.joblib")
    
    print(f"✅ Données préparées: X_train={X_train.shape}, X_test={X_test.shape}")
    return X_train, X_test, y_train, test_ids, preprocessor


if __name__ == "__main__":
    # Test du module
    print("🧪 Test du module preprocessing...")
    
    X_train, X_test, y_train, test_ids, preprocessor = prepare_train_test_data(
        sample_frac=0.1  # 10% pour test rapide
    )
    
    print(f"\nRésumé:")
    print(f"  - X_train shape: {X_train.shape}")
    print(f"  - X_test shape: {X_test.shape}")
    print(f"  - y_train distribution: {y_train.value_counts().to_dict()}")
    print(f"  - Nombre de features: {len(preprocessor.feature_names)}")
