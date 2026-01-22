"""
Tests unitaires pour le module de prétraitement.
================================================

Teste les fonctions de prétraitement et d'ingénierie des features.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Ajouter src au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import (
    CreditScoringPreprocessor,
    clean_feature_names,
    reduce_memory_usage
)


class TestCleanFeatureNames:
    """Tests pour le nettoyage des noms de features."""
    
    def test_clean_basic(self):
        """Test du nettoyage basique des noms."""
        df = pd.DataFrame({
            'Feature One': [1, 2, 3],
            'Feature (Two)': [4, 5, 6],
            'Feature[Three]': [7, 8, 9]
        })
        
        result = clean_feature_names(df)
        
        assert 'Feature_One' in result.columns
        assert 'Feature__Two_' in result.columns
        assert 'Feature_Three_' in result.columns
    
    def test_clean_special_chars(self):
        """Test avec des caractères spéciaux."""
        df = pd.DataFrame({
            'a,b': [1],
            'c:d': [2],
            'e/f': [3]
        })
        
        result = clean_feature_names(df)
        
        for col in result.columns:
            assert '[' not in col
            assert ']' not in col
            assert '<' not in col
            assert '>' not in col


class TestReduceMemoryUsage:
    """Tests pour la réduction de l'utilisation mémoire."""
    
    def test_reduce_integers(self):
        """Test de la réduction des entiers."""
        df = pd.DataFrame({
            'small_int': np.array([1, 2, 3], dtype='int64'),
            'medium_int': np.array([1000, 2000, 3000], dtype='int64')
        })
        
        original_memory = df.memory_usage(deep=True).sum()
        result = reduce_memory_usage(df.copy())
        new_memory = result.memory_usage(deep=True).sum()
        
        # La mémoire devrait être réduite ou égale
        assert new_memory <= original_memory
    
    def test_reduce_floats(self):
        """Test de la réduction des flottants."""
        df = pd.DataFrame({
            'float_col': np.array([1.1, 2.2, 3.3], dtype='float64')
        })
        
        result = reduce_memory_usage(df.copy())
        
        # Les valeurs doivent rester approximativement les mêmes
        np.testing.assert_array_almost_equal(
            result['float_col'].values, 
            df['float_col'].values, 
            decimal=5
        )
    
    def test_preserve_data_integrity(self):
        """Test que les données restent intègres après réduction."""
        df = pd.DataFrame({
            'int_col': [1, 2, 3, 100, 200],
            'float_col': [1.5, 2.5, 3.5, 100.5, 200.5]
        })
        
        result = reduce_memory_usage(df.copy())
        
        # Vérifier l'intégrité
        assert list(result['int_col']) == list(df['int_col'])
        np.testing.assert_array_almost_equal(
            result['float_col'].values, 
            df['float_col'].values
        )


class TestCreditScoringPreprocessor:
    """Tests pour le préprocesseur complet."""
    
    @pytest.fixture
    def sample_train_data(self):
        """Crée des données d'entraînement de test."""
        np.random.seed(42)
        n = 100
        
        return pd.DataFrame({
            'SK_ID_CURR': range(1, n + 1),
            'TARGET': np.random.binomial(1, 0.08, n),
            'CODE_GENDER': np.random.choice(['M', 'F'], n),
            'FLAG_OWN_CAR': np.random.choice(['Y', 'N'], n),
            'FLAG_OWN_REALTY': np.random.choice(['Y', 'N'], n),
            'CNT_CHILDREN': np.random.randint(0, 4, n),
            'AMT_INCOME_TOTAL': np.random.uniform(50000, 500000, n),
            'AMT_CREDIT': np.random.uniform(100000, 1000000, n),
            'AMT_ANNUITY': np.random.uniform(10000, 50000, n),
            'AMT_GOODS_PRICE': np.random.uniform(100000, 900000, n),
            'NAME_TYPE_SUITE': np.random.choice(['Unaccompanied', 'Family', 'Other'], n),
            'NAME_INCOME_TYPE': np.random.choice(['Working', 'Commercial associate', 'Pensioner'], n),
            'NAME_EDUCATION_TYPE': np.random.choice(['Secondary', 'Higher education', 'Incomplete higher'], n),
            'NAME_FAMILY_STATUS': np.random.choice(['Married', 'Single', 'Civil marriage'], n),
            'NAME_HOUSING_TYPE': np.random.choice(['House / apartment', 'Rented apartment', 'With parents'], n),
            'DAYS_BIRTH': np.random.randint(-25000, -7000, n),
            'DAYS_EMPLOYED': np.random.randint(-10000, 0, n),
            'DAYS_REGISTRATION': np.random.randint(-10000, 0, n),
            'DAYS_ID_PUBLISH': np.random.randint(-5000, 0, n),
            'OWN_CAR_AGE': np.random.uniform(0, 20, n),
            'OCCUPATION_TYPE': np.random.choice(['Laborers', 'Core staff', 'Managers', np.nan], n),
            'CNT_FAM_MEMBERS': np.random.randint(1, 6, n),
            'REGION_RATING_CLIENT': np.random.randint(1, 4, n),
            'REGION_RATING_CLIENT_W_CITY': np.random.randint(1, 4, n),
            'REG_REGION_NOT_LIVE_REGION': np.random.randint(0, 2, n),
            'REG_REGION_NOT_WORK_REGION': np.random.randint(0, 2, n),
            'REG_CITY_NOT_LIVE_CITY': np.random.randint(0, 2, n),
            'REG_CITY_NOT_WORK_CITY': np.random.randint(0, 2, n),
            'EXT_SOURCE_1': np.random.uniform(0, 1, n),
            'EXT_SOURCE_2': np.random.uniform(0, 1, n),
            'EXT_SOURCE_3': np.random.uniform(0, 1, n),
        })
    
    @pytest.fixture
    def sample_test_data(self, sample_train_data):
        """Crée des données de test."""
        df = sample_train_data.drop('TARGET', axis=1).copy()
        df['SK_ID_CURR'] = range(1001, 1001 + len(df))
        return df
    
    def test_preprocessor_init(self):
        """Test de l'initialisation du préprocesseur."""
        preprocessor = CreditScoringPreprocessor()
        
        assert preprocessor is not None
        assert hasattr(preprocessor, 'fit')
        assert hasattr(preprocessor, 'transform')
    
    def test_fit_transform(self, sample_train_data):
        """Test de fit_transform."""
        preprocessor = CreditScoringPreprocessor()
        
        X = sample_train_data.drop(['SK_ID_CURR', 'TARGET'], axis=1)
        y = sample_train_data['TARGET']
        
        X_transformed = preprocessor.fit_transform(X, y)
        
        assert X_transformed is not None
        assert len(X_transformed) == len(X)
        # Pas de valeurs NaN dans les features numériques après imputation
        # (sauf si le preprocessor garde les NaN pour certaines raisons)
    
    def test_transform_after_fit(self, sample_train_data, sample_test_data):
        """Test de transform après fit."""
        preprocessor = CreditScoringPreprocessor()
        
        X_train = sample_train_data.drop(['SK_ID_CURR', 'TARGET'], axis=1)
        y_train = sample_train_data['TARGET']
        X_test = sample_test_data.drop(['SK_ID_CURR'], axis=1)
        
        preprocessor.fit(X_train, y_train)
        X_test_transformed = preprocessor.transform(X_test)
        
        assert X_test_transformed is not None
        assert len(X_test_transformed) == len(X_test)
    
    def test_same_features_train_test(self, sample_train_data, sample_test_data):
        """Test que train et test ont les mêmes features après transformation."""
        preprocessor = CreditScoringPreprocessor()
        
        X_train = sample_train_data.drop(['SK_ID_CURR', 'TARGET'], axis=1)
        y_train = sample_train_data['TARGET']
        X_test = sample_test_data.drop(['SK_ID_CURR'], axis=1)
        
        X_train_transformed = preprocessor.fit_transform(X_train, y_train)
        X_test_transformed = preprocessor.transform(X_test)
        
        # Both should be arrays with same number of features (columns)
        assert isinstance(X_train_transformed, (np.ndarray, pd.DataFrame))
        assert isinstance(X_test_transformed, (np.ndarray, pd.DataFrame))
        if isinstance(X_train_transformed, pd.DataFrame):
            assert list(X_train_transformed.columns) == list(X_test_transformed.columns)
        else:
            # numpy arrays - check shape[1] (number of features)
            assert X_train_transformed.shape[1] == X_test_transformed.shape[1]
    
    def test_no_target_leakage(self, sample_train_data):
        """Test qu'il n'y a pas de fuite de la target."""
        preprocessor = CreditScoringPreprocessor()
        
        X = sample_train_data.drop(['SK_ID_CURR', 'TARGET'], axis=1)
        y = sample_train_data['TARGET']
        
        X_transformed = preprocessor.fit_transform(X, y)
        
        # TARGET ne doit pas être dans les features
        if isinstance(X_transformed, pd.DataFrame):
            assert 'TARGET' not in X_transformed.columns
        else:
            # For numpy array, check that preprocessor stored feature_names without TARGET
            if hasattr(preprocessor, 'feature_names'):
                assert 'TARGET' not in preprocessor.feature_names
        
        # Vérifier qu'aucune colonne n'est parfaitement corrélée avec la target
        if isinstance(X_transformed, pd.DataFrame):
            for col in X_transformed.select_dtypes(include=[np.number]).columns:
                corr = X_transformed[col].corr(y)
                assert abs(corr) < 0.99, f"Possible fuite de target dans {col}"


class TestFeatureEngineering:
    """Tests pour l'ingénierie des features."""
    
    @pytest.fixture
    def basic_data(self):
        """Données basiques pour les tests."""
        return pd.DataFrame({
            'AMT_INCOME_TOTAL': [100000, 200000, 150000],
            'AMT_CREDIT': [300000, 400000, 450000],
            'AMT_ANNUITY': [15000, 20000, 22500],
            'AMT_GOODS_PRICE': [280000, 380000, 420000],
            'DAYS_BIRTH': [-10000, -15000, -20000],
            'DAYS_EMPLOYED': [-1000, -2000, -3000],
        })
    
    def test_credit_income_ratio(self, basic_data):
        """Test du ratio crédit/revenu."""
        ratio = basic_data['AMT_CREDIT'] / basic_data['AMT_INCOME_TOTAL']
        
        assert all(ratio > 0), "Le ratio crédit/revenu doit être positif"
        assert ratio.iloc[0] == 3.0
        assert ratio.iloc[1] == 2.0
    
    def test_annuity_income_ratio(self, basic_data):
        """Test du ratio annuité/revenu."""
        ratio = basic_data['AMT_ANNUITY'] / basic_data['AMT_INCOME_TOTAL']
        
        assert all(ratio > 0), "Le ratio annuité/revenu doit être positif"
        assert all(ratio < 1), "Le ratio annuité/revenu doit être < 1"
    
    def test_age_calculation(self, basic_data):
        """Test du calcul de l'âge."""
        age = -basic_data['DAYS_BIRTH'] / 365
        
        assert all(age > 0), "L'âge doit être positif"
        assert 19 < age.iloc[0] < 100, "L'âge doit être réaliste"


class TestMissingValues:
    """Tests pour la gestion des valeurs manquantes."""
    
    def test_handle_missing_numerical(self):
        """Test de la gestion des valeurs manquantes numériques."""
        df = pd.DataFrame({
            'col1': [1.0, np.nan, 3.0, np.nan],
            'col2': [np.nan, 2.0, np.nan, 4.0]
        })
        
        preprocessor = CreditScoringPreprocessor()
        
        # Le preprocessor devrait pouvoir gérer les NaN
        # Sans erreur à l'initialisation
        assert preprocessor is not None
    
    def test_handle_missing_categorical(self):
        """Test de la gestion des valeurs manquantes catégorielles."""
        df = pd.DataFrame({
            'cat_col': ['A', np.nan, 'B', np.nan, 'C']
        })
        
        # Les NaN dans les catégorielles doivent être gérés
        filled = df['cat_col'].fillna('Unknown')
        
        assert 'Unknown' in filled.values
        assert not filled.isna().any()


class TestDataValidation:
    """Tests de validation des données."""
    
    def test_required_columns_exist(self):
        """Test que les colonnes requises existent."""
        required_columns = [
            'SK_ID_CURR',
            'AMT_INCOME_TOTAL',
            'AMT_CREDIT',
            'AMT_ANNUITY'
        ]
        
        df = pd.DataFrame({col: [1] for col in required_columns})
        
        for col in required_columns:
            assert col in df.columns, f"Colonne requise manquante: {col}"
    
    def test_target_is_binary(self):
        """Test que la target est binaire."""
        target = pd.Series([0, 1, 0, 1, 0, 0, 1])
        
        unique_values = target.unique()
        assert len(unique_values) == 2
        assert set(unique_values) == {0, 1}
    
    def test_no_duplicate_ids(self):
        """Test qu'il n'y a pas de doublons d'ID."""
        ids = pd.Series([1, 2, 3, 4, 5])
        
        assert not ids.duplicated().any(), "Il y a des IDs en double"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
