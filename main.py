#%%
from data.dataset import load_data
from data.dataset import check_dupe_or_null
from data.dataset import remove_outliers
from data.dataset import visualize_features
from data.dataset import visualize_features_z_score
from data.dataset import visualize_corr

from model.train import train_model

test = load_data()
check_dupe_or_null(test)
visualize_features(test)
visualize_features_z_score(test)
# After the next line is run, test will no longer contain non numerical features (can probably add back if we want)
test = remove_outliers(test)
visualize_corr(test)
# %%
