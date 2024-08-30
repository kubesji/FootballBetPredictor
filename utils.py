import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.utils import shuffle


def evaluate(prediction, bookies, results, tau=0.1, bet_split=0.05):
    # Variables for evaluation
    money_history = []
    money, bets, wins = 1, 0, 0

    # Iterate through the t
    for p, b, r in zip(prediction, bookies, results):
        position = np.argmin(p)
        ratio = (b[position] / p[position]) - 1
        if ratio > tau:
            bet = money * bet_split
            money -= bet
            bets += 1
            if r == position:
                wins += 1
                money += bet * b[position]
        money_history.append(money)

    print(f"# Games: {len(money_history)}, # Bets: {bets}, # Wins: {wins}, money {money}")
    # let us make a simple graph
    fig = plt.figure(figsize=[7, 5])
    ax = plt.subplot(111)

    # set the basic properties
    ax.set_xlabel('Game number')
    ax.set_ylabel('Cash increase')
    ax.set_title('Earnings on bets')
    ax.title.set_weight('bold')

    # change the color of the top and right spines to opaque gray
    ax.spines['right'].set_color((.8, .8, .8))
    ax.spines['top'].set_color((.8, .8, .8))

    # set the limits
    ax.set_xlim(0, math.ceil(len(money_history) / 100) * 100)
    ax.set_ylim(0, math.ceil(max(money_history) / 5) * 5)

    # set the grid on
    ax.grid(True, linestyle='--')
    ax.plot(money_history)

    # plt.yscale("log")
    plt.show()


def generate_datasets(n_features, shuffle_ds=True, random_state=42):
    # Doesn't provide any useful information
    cols_drop = ['result', 'league_id', 'season_id', 'season_name_codes']
    # We want to maximise profit, therefore bet are placed on highest odds
    odds_columns = ['3W__X_max', '3W__1_max', '3W__2_max']
    # Venue Sub-dataset
    venue_file = pd.read_csv('data/Modeling_Final/1_venue.csv').set_index('id')
    print(f'venue_file loaded: {venue_file.shape}')
    # Standings Sub-dataset
    standings_file = pd.read_csv('data/Modeling_Final/2_standings.csv').set_index('id').drop(columns=cols_drop)
    print(f'standings_file loaded: {standings_file.shape}')
    # Form and Rest Sub-dataset
    form_file = pd.read_csv('data/Modeling_Final/3_form_rest.csv').set_index('id').drop(columns=cols_drop)
    print(f'form_file loaded: {form_file.shape}')
    # Stats Sub-dataset
    stats_file = pd.read_csv('data/Modeling_Final/4_stats.csv').set_index('id').drop(columns=cols_drop)
    print(f'stats_file loaded: {stats_file.shape}')
    # Odds Sub-dataset - not used in classification -> only for evaluation
    odds_file = pd.read_csv('data/Modeling_Final/5_odds.csv').set_index('id')
    odds_file = odds_file[odds_columns]  # Get odds for evaluation, discard rest
    print(f'odds_file loaded: {odds_file.shape}')

    # Merge sub-datasets
    # Odds merged with the rest to match any row drops
    complete_df = venue_file.merge(standings_file, left_index=True, right_index=True) \
        .merge(form_file, left_index=True, right_index=True) \
        .merge(stats_file, left_index=True, right_index=True) \
        .merge(odds_file, left_index=True, right_index=True)

    # Drop NA
    complete_df.dropna(inplace=True)
    # Categorical columns
    categorical_columns = ['result', 'colors_home_color', 'colors_away_color', 'league_id', 'season_id', 'venue_id',
                           'venue_city', 'season_name_codes', 'home_country_id', 'venue_surface_isgrass', 'night_game',
                           'home_not_home', 'travel_outside_state']
    rank_cols = complete_df.filter(regex='rank').columns.tolist() + ['ROUND']
    complete_df[categorical_columns + rank_cols] = complete_df[categorical_columns + rank_cols].astype('category')
    categorical_columns.remove('result')
    # Numeric columns
    float_cols = complete_df.select_dtypes(np.float_).columns.tolist()
    int_cols = complete_df.select_dtypes(np.int_).columns.tolist()
    numeric_columns = float_cols + int_cols

    # Now remove column names with odds - they won't be in the dataset
    for oc in odds_columns:
        numeric_columns.remove(oc)

    complete_df.loc[:, float_cols] = complete_df.loc[:, float_cols].apply(pd.to_numeric, downcast='float')
    complete_df.loc[:, int_cols] = complete_df.loc[:, int_cols].apply(pd.to_numeric, downcast='integer')

    print('DataFrame shape: ', complete_df.shape)

    # Drop odds from dataset
    odds = complete_df[odds_columns]
    complete_df = complete_df.drop(columns=odds_columns)

    # Split training/test sets
    X = complete_df.loc[:, 'league_id':]
    y = complete_df['result']
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=None, stratify=y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=random_state,
                                                        shuffle=shuffle_ds, stratify=(y if shuffle_ds else None))
    # Odds needed only for test
    odds_test = np.array(odds.loc[y_test.index, :])

    # Sets shape
    print('X_train shape: ', X_train.shape, '. Y train shape: ', y_train.shape,
          '\nX_test shape: ', X_test.shape, '. Y test shape: ', y_test.shape)

    # Pre-processing methods
    full_pipeline_mms = ColumnTransformer([
        ('num', StandardScaler(), numeric_columns),
        ('cat_hot', OneHotEncoder(handle_unknown='ignore'), categorical_columns),
        ('rank', OrdinalEncoder(), rank_cols)
    ], remainder='passthrough')
    X_train_transformed = full_pipeline_mms.fit_transform(X_train)
    X_test_transformed = full_pipeline_mms.transform(X_test)

    print('X_train_transformed shape: ', X_train_transformed.shape,
          '\nX_test_transformed shape: ', X_test_transformed.shape)

    # Feature selection
    fs = SelectKBest(k=n_features, score_func=f_classif)
    X_train_transformed_reduced = fs.fit_transform(X_train_transformed, y_train)
    X_test_transformed_reduced = fs.transform(X_test_transformed)

    return X_train_transformed_reduced, y_train, X_test_transformed_reduced, y_test, odds_test
