import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

from sklearn.decomposition import PCA

from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

def train_rf(X, y, validation_data):

    param_grid = {
        "rf__n_estimators": [200, 500, 1000],
        "rf__max_depth": [None, 10, 20, 40],
        "rf__min_samples_split": [2, 5, 10],
        "rf__min_samples_leaf": [1, 2, 4, 8],
        "rf__max_features": ["sqrt", "log2", 0.5],
        "rf__class_weight": [None, "balanced"],
        # Optional: tune imputer too
        # "imputer__n_neighbors": [3, 5, 7, 11],
        # "imputer__weights": ["uniform", "distance"],
    }
    
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

    pipe = Pipeline(
        steps=[
            ("imputer", KNNImputer(n_neighbors=5, weights="distance")),
            ("rf", RandomForestClassifier(random_state=42)),
        ]
    )

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_grid,
        cv=cv,
        n_iter=30,
        scoring="accuracy",
        n_jobs=-1,
        error_score="raise",
        verbose=0,
        random_state=42,
    )


    search.fit(X, y)
    best_model = search.best_estimator_
    
    validation_X = validation_data[X.columns]
    validation_y = validation_data[y.name]
    y_pred = best_model.predict(validation_X)
    accuracy = accuracy_score(validation_y, y_pred)
    
    return accuracy

def get_pca_explained_variance(data, n_components=2, standardize=True):
    """
    Run PCA on a DataFrame and return the explained variance by each component.
    
    Parameters:
    -----------
    data : pandas DataFrame
        DataFrame containing numerical data for PCA analysis
    n_components : int, default 2
        Number of principal components to compute
    standardize : bool, default True
        Whether to standardize the data before PCA

    Returns:
    --------
    dict
        Dictionary containing:
        - 'explained_variance_ratio': array of explained variance ratio for each component
        - 'cumulative_variance': cumulative explained variance
        - 'total_variance': total variance explained by all components
        - 'variance_by_component': dict mapping component number to variance percentage
    """

    
    # Handle missing values
    if data.isna().any().any():
        imputer = KNNImputer(n_neighbors=5, weights="distance")
        processed_data = pd.DataFrame(
            imputer.fit_transform(data),
            columns=data.columns
        )
    else:
        processed_data = data
    
    # Standardize if requested
    if standardize:
        processed_data = (processed_data - processed_data.mean()) / processed_data.std()
    
    # Apply PCA
    pca = PCA(n_components=min(n_components, processed_data.shape[1]))
    pca.fit(processed_data)
    
    # Prepare results
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_) * 100
    variance_by_component = {
        f"PC{i+1}": var * 100 
        for i, var in enumerate(pca.explained_variance_ratio_)
    }
    
    return {
        'explained_variance_ratio': pca.explained_variance_ratio_ * 100,
        'cumulative_variance': cumulative_variance,
        'total_variance': np.sum(pca.explained_variance_ratio_) * 100,
        'variance_by_component': variance_by_component,
        'pca_model': pca
    }

def identify_significant_proteins(data, condition_col='Condition', significance_level=0.05, print_results=True,
                                  require_all_significant=False):
    """
    Identify proteins with significant differences between conditions using t-tests.

    Parameters:
    -----------
    data : pandas DataFrame
        DataFrame containing protein abundance data with proteins as columns
        and a column indicating conditions
    condition_col : str, default 'Condition'
        Name of the column containing condition labels
    significance_level : float, default 0.05
        Threshold p-value for determining significance
    print_results : bool, default True
        Whether to print the results
    require_all_significant : bool, default False
        If True, proteins must be significant across all conditions
        If False, proteins are considered significant if they differ in at least one condition

    Returns:
    --------
    tuple : (significant_proteins, not_significant_proteins, t_test_results)
        Lists of significant and non-significant protein names, and DataFrame with all p-values
    """
    import pandas as pd
    from scipy.stats import ttest_ind

    # Get protein columns (excluding the condition column)
    protein_cols = [col for col in data.columns if col != condition_col]

    # Identify conditions in the dataset
    conditions = data[condition_col].unique()

    # Create a DataFrame to store t-test results
    t_test_results = pd.DataFrame(index=protein_cols)

    # Perform t-tests for each protein between each condition and all others
    for condition in conditions:
        # Select data for the current condition vs other conditions
        condition_data = data.loc[data[condition_col] == condition, protein_cols]
        other_data = data.loc[data[condition_col] != condition, protein_cols]

        # Initialize p-value list
        p_values = []

        # Perform t-test for each protein between conditions
        for protein in protein_cols:
            # Skip if the protein has all NaN values in either group
            if condition_data[protein].isna().all() or other_data[protein].isna().all():
                p_values.append(1.0)  # Not significant if all values are missing
                continue

            # Calculate t-test (handling potential empty arrays)
            try:
                _, p_value = ttest_ind(
                    condition_data[protein].dropna(),
                    other_data[protein].dropna(),
                    equal_var=False  # Using Welch's t-test which doesn't assume equal variances
                )
                p_values.append(p_value)
            except:
                # If t-test fails (e.g., only one sample), assign non-significant p-value
                p_values.append(1.0)

        # Assign p-values to the results DataFrame
        t_test_results[f"{condition}_p_value"] = p_values

    # Identify significant proteins based on the selected approach
    if require_all_significant:
        # Must be significant in all conditions
        significant_mask = t_test_results.le(significance_level).all(axis=1)
    else:
        # Significant in at least one condition
        significant_mask = t_test_results.le(significance_level).any(axis=1)

    significant_proteins = t_test_results[significant_mask].index.tolist()
    not_significant_proteins = t_test_results[~significant_mask].index.tolist()

    # Display the results if requested
    if print_results:
        print(f"Proteins with Significant Differences ({len(significant_proteins)}):")
        print(significant_proteins)
        print(f"\nProteins with Non-Significant Differences ({len(not_significant_proteins)}):")
        print(not_significant_proteins)

    return significant_proteins, not_significant_proteins, t_test_results


def _bh_adjust(p_values):
    """Benjamini-Hochberg FDR adjustment. NaN p-values are ignored and stay NaN."""
    import numpy as np

    p = np.asarray(p_values, dtype=float)
    adjusted = np.full(p.shape, np.nan)

    observed = ~np.isnan(p)
    m = int(observed.sum())
    if m == 0:
        return adjusted

    p_obs = p[observed]
    order = np.argsort(p_obs)
    ranked = p_obs[order] * m / np.arange(1, m + 1)

    # Enforce monotonicity from the largest p-value down, then cap at 1
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    ranked = np.clip(ranked, 0., 1.)

    unsorted = np.empty(m, dtype=float)
    unsorted[order] = ranked
    adjusted[observed] = unsorted

    return adjusted


def run_ttest(data, condition_col='Condition', equal_var=False,
              adjust_pvalues=True, significance_level=0.05, min_samples=2,
              sort_by_pvalue=True):
    """
    Run two-sample t-tests between the two conditions present in a DataFrame.

    Parameters:
    -----------
    data : pandas DataFrame
        DataFrame with one protein per column plus a single column holding the
        condition label. Every column other than `condition_col` is tested, and
        `condition_col` must contain exactly two distinct values.
    condition_col : str, default 'Condition'
        Name of the column containing condition labels.
    equal_var : bool, default False
        If False (default) run Welch's t-test, which does not assume equal
        variances. If True, run the standard pooled-variance t-test.
    adjust_pvalues : bool, default True
        If True, add a Benjamini-Hochberg adjusted p-value column and use it to
        call significance. If False, significance is called on the raw p-value.
    significance_level : float, default 0.05
        Threshold used for the `significant` column.
    min_samples : int, default 2
        Minimum number of non-missing observations required in each group. Proteins
        below this in either group get NaN statistics and are excluded from the
        BH adjustment.
    sort_by_pvalue : bool, default True
        Whether to sort the returned table by ascending p-value.

    Returns:
    --------
    pandas DataFrame
        One row per protein with columns:
        - 'Protein'
        - 'n_<cond1>', 'n_<cond2>' : non-missing sample counts
        - 'mean_<cond1>', 'mean_<cond2>', 'sd_<cond1>', 'sd_<cond2>'
        - 'diff' : mean(cond2) - mean(cond1) (a log2 fold change for log2 input)
        - 'statistic', 'pvalue'
        - 'adj_pvalue' : BH adjusted p-value (only if adjust_pvalues=True)
        - 'significant' : bool
    """
    import pandas as pd
    import numpy as np
    from scipy.stats import ttest_ind

    if condition_col not in data.columns:
        raise ValueError(f"Condition column '{condition_col}' not found in data")

    conditions = pd.Series(data[condition_col].dropna().unique()).sort_values().tolist()
    if len(conditions) != 2:
        raise ValueError(
            f"Expected exactly 2 distinct values in '{condition_col}', found "
            f"{len(conditions)}: {conditions}"
        )
    cond1, cond2 = conditions

    # Every column other than the condition column is a protein
    protein_cols = [col for col in data.columns if col != condition_col]
    if len(protein_cols) == 0:
        raise ValueError("No protein columns available to test")

    non_numeric = [
        col for col in protein_cols
        if not pd.api.types.is_numeric_dtype(data[col])
    ]
    if len(non_numeric) > 0:
        raise ValueError(
            f"Expected all columns other than '{condition_col}' to be numeric "
            f"protein abundances, but found non-numeric columns: {non_numeric}"
        )

    group1 = data.loc[data[condition_col] == cond1, protein_cols]
    group2 = data.loc[data[condition_col] == cond2, protein_cols]

    rows = list()
    for protein in protein_cols:
        x1 = group1[protein].dropna().values
        x2 = group2[protein].dropna().values

        if len(x1) >= min_samples and len(x2) >= min_samples:
            statistic, p_value = ttest_ind(x2, x1, equal_var=equal_var)
        else:
            statistic, p_value = np.nan, np.nan

        rows.append({
            "Protein": protein,
            f"n_{cond1}": len(x1),
            f"n_{cond2}": len(x2),
            f"mean_{cond1}": np.mean(x1) if len(x1) > 0 else np.nan,
            f"mean_{cond2}": np.mean(x2) if len(x2) > 0 else np.nan,
            f"sd_{cond1}": np.std(x1, ddof=1) if len(x1) > 1 else np.nan,
            f"sd_{cond2}": np.std(x2, ddof=1) if len(x2) > 1 else np.nan,
            "diff": (np.mean(x2) - np.mean(x1)) if (len(x1) > 0 and len(x2) > 0) else np.nan,
            "statistic": statistic,
            "pvalue": p_value,
        })

    results = pd.DataFrame(rows)

    if adjust_pvalues:
        results["adj_pvalue"] = _bh_adjust(results["pvalue"].values)
        results["significant"] = results["adj_pvalue"] < significance_level
    else:
        results["significant"] = results["pvalue"] < significance_level

    if sort_by_pvalue:
        results = results.sort_values("pvalue", na_position="last").reset_index(drop=True)

    return results


def calculate_percentage_of_variation(data, size, n_iterations=100, n_components=2,
                                      condition_column=None, impute_strategy='mean',
                                      standardize=True, random_state=None, verbose=False,
                                      return_all_stats=False):
    """
    Calculate the percentage of variation explained by principal components across multiple
    bootstrapped samples of a dataset.

    Parameters:
    -----------
    data : pandas DataFrame
        DataFrame containing protein abundance data with proteins as columns
    size : int
        Sample size to use for each bootstrap iteration
    n_iterations : int, default 100
        Number of bootstrap iterations to perform
    n_components : int, default 2
        Number of principal components to use
    condition_column : str, default None
        If provided, sampling will be done separately within each condition
    impute_strategy : str, default 'mean'
        Strategy for SimpleImputer ('mean', 'median', 'most_frequent', 'constant')
    standardize : bool, default True
        Whether to standardize the data before PCA
    random_state : int, default None
        Random seed for reproducibility
    verbose : bool, default False
        Whether to print progress information
    return_all_stats : bool, default False
        If True, returns all variation values across iterations

    Returns:
    --------
    dict or tuple
        If return_all_stats=True: Dictionary with statistics for each condition (or 'all')
        If return_all_stats=False: Tuple of (average, minimum, maximum) variation percentages
    """
    import pandas as pd
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer

    # Set random seed if provided
    if random_state is not None:
        np.random.seed(random_state)

    # If condition column is provided, process each condition separately
    if condition_column is not None and condition_column in data.columns:
        conditions = data[condition_column].unique()
        results = {}

        for condition in conditions:
            if verbose:
                print(f"Processing condition: {condition}")

            # Filter data for the current condition
            condition_data = data[data[condition_column] == condition].drop(columns=[condition_column])

            # Calculate variation for this condition
            if size > len(condition_data):
                if verbose:
                    print(
                        f"Warning: Sample size {size} is larger than available data for condition {condition} ({len(condition_data)})")
                condition_size = len(condition_data)
            else:
                condition_size = size

            condition_results = _calculate_variation_for_dataset(
                condition_data, condition_size, n_iterations, n_components,
                impute_strategy, standardize, verbose
            )
            results[condition] = condition_results

        return results
    else:
        # Process the entire dataset without considering conditions
        if condition_column is not None and condition_column not in data.columns:
            if verbose:
                print(f"Warning: Condition column '{condition_column}' not found in data")

        # If condition column exists but we're not using it, drop it
        if condition_column in data.columns:
            processed_data = data.drop(columns=[condition_column])
        else:
            processed_data = data

        # Calculate variation for the whole dataset
        return _calculate_variation_for_dataset(
            processed_data, size, n_iterations, n_components,
            impute_strategy, standardize, verbose, return_all_stats
        )


def _calculate_variation_for_dataset(data, size, n_iterations, n_components,
                                     impute_strategy, standardize, verbose, return_all_stats=False):
    """Helper function to calculate variation for a single dataset"""
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer

    all_variations = []

    # Check if sample size is larger than dataset
    if size > len(data):
        if verbose:
            print(f"Warning: Sample size {size} is larger than available data ({len(data)})")
        size = len(data)

    for i in range(n_iterations):
        if verbose and i % 10 == 0:
            print(f"  Iteration {i}/{n_iterations}")

        try:
            # Take a random sample with replacement
            sampled_data = data.sample(n=size, replace=True)

            # Handle missing values if present
            if sampled_data.isna().any().any():
                if verbose and i == 0:
                    print("  Imputing missing values")

                # Impute missing values
                imputer = SimpleImputer(strategy=impute_strategy)
                sampled_data_imputed = pd.DataFrame(
                    imputer.fit_transform(sampled_data),
                    columns=sampled_data.columns
                )

                # Use imputed data for further processing
                processed_data = sampled_data_imputed
            else:
                processed_data = sampled_data

            # Remove columns with zero variance
            non_zero_var_cols = processed_data.columns[processed_data.std() != 0]

            if len(non_zero_var_cols) < 2:
                if verbose:
                    print(
                        f"  Warning: Found only {len(non_zero_var_cols)} columns with non-zero variance. Skipping iteration.")
                continue

            data_for_pca = processed_data[non_zero_var_cols]

            # Standardize if requested
            if standardize:
                data_for_pca = (data_for_pca - data_for_pca.mean()) / data_for_pca.std()

            # Apply PCA
            pca = PCA(n_components=min(n_components, len(non_zero_var_cols)))
            pca.fit(data_for_pca)

            # Calculate percentage of variation
            percentage_variation = sum(pca.explained_variance_ratio_) * 100
            all_variations.append(percentage_variation)

        except Exception as e:
            if verbose:
                print(f"  Error in iteration {i}: {str(e)}")

    # Calculate statistics
    if len(all_variations) == 0:
        if verbose:
            print("No valid iterations completed")
        return (0, 0, 0, 0, []) if return_all_stats else (0, 0, 0)

    avg_var = np.mean(all_variations)
    min_var = np.min(all_variations)
    max_var = np.max(all_variations)
    std_var = np.std(all_variations)

    if return_all_stats:
        return avg_var, min_var, max_var, std_var, all_variations
    else:
        return avg_var, min_var, max_var