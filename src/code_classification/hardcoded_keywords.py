def hardcoded_keywords():
    """
    Function to populate a dictionary with hardcoded keywords based on their functionality.

    :return (dict): dictionary with hardcoded keywords
    """

    hardcoded_keywords = {
        "Environment": {
            "PL_keywords": ["import ", "requirements.txt", "environment.yml", "!pip install", "!conda install", "virtualenv", "venv", "docker-compose", "pip3 install", "conda install", "install.packages", "setup environment"],
            "NL_keywords": ["environment setup", "package installation", "dependency management", "virtual environment", "setup instructions", "installation guide", "libraries", "import packages", "import libraries"],
            "PL_keywords_weak": ["install", "setup.py", "pipenv", "pyenv", "sdk", "package.json", "Gemfile", "requirements", "dependency"],
            "NL_keywords_weak": ["setup environment", "installing dependencies", "manage packages", "setup guide", "configure", "environment configuration", "software installation", "installing software", "setup steps"],
            #"PL_keywords_very_weak": ["import ", "library", "package", "module", "install", "require", "setup.py", "pip install", "conda install", "importlib", "conda", "pip", "npm", "brew", "apt-get", "yum", "installing", "setting up", "configure", "setup", "environment", "dependencies", "packages", "library", "import", "install", "requirements", "setup.py", "requirements.txt", "environment.yml", "virtualenv", "venv", "docker-compose", "pip3 install", "conda install", "install.packages", "setup environment"], 
            #"NL_keywords_very_weak": ["setup", "configuration", "dependencies", "environment setup", "install packages", "setup environment", "library versions", "dependency management", "environment setup", "package installation", "dependency management", "virtual environment", "setup instructions", "installation guide", "libraries", "import packages", "import libraries", "install", "setup.py", "pipenv", "pyenv", "sdk", "package.json", "Gemfile", "requirements", "dependency", "setup environment", "installing dependencies", "manage packages", "setup guide", "configure", "environment configuration", "software installation", "installing software", "setup steps"],
        
        },
        "Data_Extraction": {
            "PL_keywords": ["read_csv", "read_excel", "read_sql", "pd.read_", "BeautifulSoup", "scrapy.Spider", "API call", "load_dataset", "pd.read_json", "pd.read_html", "pd.read_"],
            "NL_keywords": ["data extraction", "extraction data", "data loading", "web scraping", "API request", "reading datasets", "data import", "load data", "load the data"],
            "PL_keywords_weak": ["fetch", "query", "open", "requests.get", "urllib.request", "pandasql", "read_table", "dataframe.read", "extract"],
            "NL_keywords_weak": ["getting data", "data retrieval", "fetching data", "data access", "collecting data", "data acquisition", "accessing web data", "downloading data", "importing data"],
            #"PL_keywords_very_weak": ["read_csv", "read_excel", "load", "fetch", "API", "scrape", "DataFrame", "read_json", "pd.read_sql", "open", "requests.get", "read_sql", "pd.read_", "BeautifulSoup", "scrapy.Spider", "API call", "load_dataset", "pd.read_json", "pd.read_html", "pd.read_", "fetch", "query", "open", "requests.get", "urllib.request", "pandasql", "read_table", "dataframe.read", "extract"],
            #"NL_keywords_very_weak": ["load data", "read data", "import data", "extract data", "source data", "retrieve data", "data acquisition", "fetch data", "API data", "data extraction", "extraction data", "data loading", "web scraping", "API request", "reading datasets", "data import", "load data", "load the data", "getting data", "data retrieval", "fetching data", "data access", "collecting data", "data acquisition", "accessing web data", "downloading data", "importing data"],
        },
        "Exploratory_Data_Analysis": {
            "PL_keywords": [".describe(", ".info(", "value_counts()", "df.isnull().sum", "sns.pairplot", ".corr", ".head(", ".tail(", "plt.plot", "df.hist", "sns.scatterplot", "sns.heatmap", "plt.boxplot", "fig, ax = plt.subplots", "plt.bar", ".hist(", ".plot(", ".scatter(", ".boxplot(", ".heatmap(", ".bar("],
            "NL_keywords": ["plot data",  "data plot", "data visualization", "visualization data", "exploratory data analysis", "data exploration", "exploration data", "data distribution", "missing values", "statistical summary", "initial observations", "data overview", "duplicates", "outliers"],
            "PL_keywords_weak": [".mean(", ".std(", ".plot(", ".histogram(", "plotly", "matplotlib", ".boxplot(", "scatter_matrix", "pairgrid", "distplot", "sns.", "seaborn."],
            "NL_keywords_weak": ["data insights", "analyze data", "data trends", "data patterns", "visual analysis", "summary statistics", "data exploration", "analysis report", "data properties", "data characteristics", "data patterns", "data distribution", "data summary"],
            #"PL_keywords_very_weak": ["head", "describe", "info", "histogram", "boxplot", ".boxplot(", "value_counts", "plt.show", "df.plot", "sns.pairplot", "sns.heatmap", "sns.countplot", "sns.distplot", "sns.scatterplot", "sns.lineplot", "sns.boxplot", "sns.violinplot", "sns.swarmplot", "sns.jointplot", "sns.lmplot", "sns.catplot", "sns.relplot", "sns.regplot", "sns.residplot", "sns.kdeplot", "sns.ecdfplot", "sns.barplot", "sns.pointplot", "sns.stripplot", "sns.boxenplot", "sns.violinplot", "sns.swarmplot", "sns.catplot", "sns.pairplot", "sns.jointplot", "sns.lmplot", "sns.relplot", "sns.scatterplot", "sns.lineplot", "sns.histplot", "sns.kdeplot", "sns.ecdfplot", "sns.rugplot", "sns.barplot", "sns.countplot", "sns.pointplot", "sns.violinplot", "sns.boxenplot", "sns.stripplot", "sns.swarmplot", "sns.violinplot", "sns.catplot", "sns.pairplot", "sns.jointplot", "sns.lmplot", "sns.relplot", "sns.scatterplot", "sns.lineplot", "sns.histplot", "sns.kdeplot", "sns.ecdfplot", "sns.rugplot", "sns.barplot", "sns.countplot", "sns.pointplot", "sns.violinplot", "sns.boxenplot", "sns.stripplot", "sns.swarmplot", "sns.violinplot", "sns.catplot", "sns.pairplot", "sns.jointplot", "sns.lmplot", "sns.relplot", "sns.scatterplot", "sns.lineplot", "sns.histplot", "sns.kdeplot", "sns.ecdfplot", "sns.rugplot", "sns.barplot", "sns.countplot", "sns.pointplot", "sns.violinplot", "sns.boxenplot", "sns.stripplot", "sns.swarmplot", "sns.violinplot", "sns.catplot", "sns.pairplot", "sns.jointplot", "sns.lmplot", "sns.relplot", "sns.scatterplot", "sns.lineplot", "sns.histplot", "sns.kdeplot", "sns.ecdfplot", "sns.rugplot", "sns.barplot", "sns.countplot", "sns.pointplot", "sns.violinplot", "sns.boxenplot", "sns.stripplot", "sns.swarmplot", "sns.violinplot", "sns.catplot", "sns.pairplot", "sns.jointplot", "sns.lmplot", "sns.relplot", "sns.scatterplot", "sns.lineplot", "sns.histplot", "sns.kdeplot", "sns.ecdfplot", "sns.rugplot", "sns.barplot", "sns.countplot", "sns.pointplot", "sns.violinplot", "sns.boxenplot", "sns.stripplot", "sns.swarmplot", "sns.violinplot", "sns.catplot", "sns.pairplot", "sns.jointplot", "sns.lmplot", "sns.relplot", "sns.scatterplot", "sns.lineplot", "sns.histplot", "sns.kdeplot", "sns.ecdfplot", "sns.rugplot", "sns.barplot", "sns.countplot", "sns.pointplot", "sns.violinplot", "sns.boxenplot", "sns.stripplot", "sns.swarmplot", "sns.violinplot", "sns.catplot", "sns.pairplot", "sns.jointplot", "sns.lmplot", "sns.relplot", "sns.scatterplot", "sns.lineplot", "sns.histplot", "sns.kdeplot", "sns.ecdfplot", "sns.rugplot", "sns.barplot", "sns.countplot", "sns.pointplot", "sns.violinplot", "sns.boxenplot", "sns.stripplot", "sns.swarmplot", "sns.violinplot", "sns.catplot", "sns.pairplot", "sns.jointplot", "sns.lmplot", "sns.relplot", "sns.scatterplot", "sns.lineplot", "sns.histplot", "sns.kdeplot", "sns.ecdfplot", "sns.rugplot", "sns.barplot", "sns.countplot", "sns.pointplot", "sns.violinplot", "sns.boxenplot", "sns.stripplot", "sns.swarmplot", "sns.violinplot", "sns.catplot", "sns.pairplot", "sns.jointplot", "sns.lmplot", "sns.relplot", "sns.scatter", "sns.heatmap", "plt.boxplot", "fig, ax = plt.subplots", "plt.bar", ".hist(", ".plot(", ".scatter(", ".boxplot(", ".heatmap(", ".bar("],
            #"NL_keywords_very_weak": ["summary statistics", "distribution", "data exploration", "initial analysis", "explore data", "data review", "first look", "data quality assessment", "data overview", "data summary", "data insights", "analyze data", "data trends", "data patterns", "visual analysis", "summary statistics", "data exploration", "analysis report", "data properties", "data characteristics", "data patterns", "data distribution", "data summary", "head", "describe", "info", "histogram", "boxplot", ".boxplot(", "value_counts", "plt.show", "df.plot", "sns.pairplot", "sns.heatmap", "sns.countplot", "sns.distplot", "sns.scatterplot", "sns.lineplot", "sns.boxplot", "sns.violinplot", "sns.swarmplot", "sns.jointplot", "sns.lmplot", "sns.catplot", "sns.relplot", "sns.regplot", "sns.residplot", "sns.kdeplot", "sns.ecdfplot", "sns.barplot", "sns.pointplot", "sns.stripplot", "sns.boxenplot", "sns.violinplot", "sns.swarmplot", "sns.catplot", "sns.pairplot", "sns.jointplot", "sns.lmplot", "sns.relplot", "sns.scatterplot", "sns.lineplot", "sns.histplot", "sns.kdeplot", "sns.ecdfplot", "sns.rugplot", "sns.barplot", "sns.countplot", "sns.pointplot", "sns.violinplot", "sns.boxenplot", "sns.stripplot", "sns.swarmplot", "sns.violinplot", "sns.catplot", "sns.pairplot", "sns.jointplot", "sns.lmplot", "sns.relplot", "sns.scatterplot", "sns.lineplot", "sns.histplot", "sns.kdeplot", "sns.ecdfplot", "sns.rugplot", "sns.barplot", "sns.countplot", "sns.pointplot", "sns.violinplot", "sns.boxenplot", "sns.stripplot", "sns.swarmplot", "sns.violinplot", "sns.catplot", "sns.pairplot", "sns.jointplot", "sns.lmplot", "sns.relplot", "sns.scatterplot", "sns.lineplot", "sns.histplot", "sns.kdeplot", "sns.ecdfplot", "sns.rugplot", "sns.barplot", "sns.countplot", "sns.pointplot", "sns.violinplot", "sns.boxenplot", "sns.stripplot", "sns.swarmplot", "sns.violinplot", "sns.catplot", "sns.pairplot", "sns.jointplot", "sns.lmplot"],
        },
        "Data_Transform": {
            "PL_keywords": [".merge(", ".join(", ".concat(", ".pivot(", ".groupby(", ".fillna(", ".dropna(", ".apply(", ".map(", ".replace(", ".to_datetime(", ".to_numeric("],
            "NL_keywords": ["data transformation", "transformation data", "data transforming", "transforming data", "data cleaning", "cleaning data", "feature engineering", "data manipulation", "transforming data", "data normalization", "data preprocessing", "data transformation"],
            "PL_keywords_weak": ["fit_transform", "transform", ".astype(", "pandas.get_dummies", ".cut(", ".qcut(", "DataFrameMapper", "ColumnTransformer", ".normalize(", ".standardize(", "MinMaxScaler", "LabelEncoder"],
            "NL_keywords_weak": ["altering data", "data adjustment", "modifying data", "data changes", "data conversion", "data format change", "data reshaping", "data restructuring", "data alteration"],
            #"PL_keywords_very_weak": ["merge", "join", "concatenate", "groupby", "pivot", "filter", "df.apply", "replace", "pd.melt", "pd.pivot_table", "pd.concat", "pd.merge", "pd.join", "pd.groupby", "pd.fillna", "pd.dropna", "pd.apply", "pd.map", "pd.replace", "pd.to_datetime", "pd.to_numeric", "astype", "pandas.get_dummies", "cut", "qcut", "DataFrameMapper", "ColumnTransformer", "normalize", "standardize", "MinMaxScaler", "LabelEncoder", "merge", "join", "concatenate", "groupby", "pivot", "filter", "map", "df.apply", "replace", "pd.melt", "pd.pivot_table", "pd.concat", "pd.merge", "pd.join", "pd.groupby", "pd.fillna", "pd.dropna", "pd.apply", "pd.map", "pd.replace", "pd.to_datetime", "pd.to_numeric", "astype", "pandas.get_dummies", "cut", "qcut", "DataFrameMapper", "ColumnTransformer", "normalize", "standardize", "MinMaxScaler", "LabelEncoder", "merge", "join", "concatenate", "groupby", "pivot", "filter", "map", "df.apply", "replace", "pd.melt", "pd.pivot_table", "pd.concat", "pd.merge", "pd.join", "pd.groupby", "pd.fillna", "pd.dropna", "pd.apply", "pd.map", "pd.replace", "pd.to_datetime", "pd.to_numeric", "astype", "pandas.get_dummies", "cut", "qcut", "DataFrameMapper", "ColumnTransformer", "normalize", "standardize", "MinMaxScaler", "LabelEncoder", "merge", "join", "concatenate", "groupby", "pivot", "filter", "map", "df.apply", "replace", "pd.melt", "pd.pivot_table", "pd.concat", "pd.merge", "pd.join", "pd.groupby", "pd.fillna", "pd.dropna", "pd.apply", "pd.map", "pd.replace", "pd.to_datetime", "pd.to_numeric", "astype", "pandas.get_dummies", "cut", "qcut", "DataFrameMapper", "ColumnTransformer", "normalize", "standardize", "MinMaxScaler", "LabelEncoder", "merge", "join", "concatenate", "groupby", "pivot", "filter", "map", "df.apply", "replace", "pd.melt", "pd.pivot_table", "pd.concat", "pd.merge", "pd.join", "pd.groupby", "pd.fillna", "pd.dropna", "pd.apply", "pd.map", "pd.replace", "pd.to_datetime", "pd.to_numeric", "astype", "pandas.get_dummies", "cut", "qcut", "DataFrameMapper", "ColumnTransformer", "normalize", "standardize", "MinMaxScaler", "LabelEncoder", "merge", "join", "concatenate", "groupby", "pivot", "filter", "map", "df.apply", "replace", "pd.melt", "pd.pivot_table", "pd.concat", "pd.merge", "pd.join", "pd.groupby", "pd.fillna", "pd.dropna", "pd.apply", "pd.map", "pd.replace", "pd.to_datetime", "pd.to_numeric", "astype", "pandas.get_dummies", "cut", "qcut", "DataFrameMapper", "ColumnTransformer", "normalize", "standardize", "MinMaxScaler", "LabelEncoder", "merge", "join", "concatenate", "groupby", "pivot", "filter", "map", "df.apply", "replace", "pd.melt", "pd.pivot_table", "pd.concat", "pd.merge", "pd.join", "pd.groupby", "pd.fillna", "pd.dropna", "pd.apply", "pd.map", "pd.replace", "pd.to_datetime", "pd.to_numeric", "astype", "pandas.get_dummies", "cut", "qcut", "DataFrameMapper", "ColumnTransformer", "normalize", "standardize", "MinMaxScaler", "LabelEncoder", "merge", "join", "concatenate", "groupby", "pivot", "filter", "map", "df.apply", "replace", "pd.melt", "pd.pivot_table", "pd.concat", "pd.merge", "pd.join", "pd.groupby", "pd.fillna", "pd.dropna", "pd.apply", "pd.map", "pd.replace", "pd.to_datetime", "pd.to_numeric", "astype", "pandas.get_dummies", "cut", "qcut", "DataFrameMapper", "ColumnTransformer", "normalize"],
            #"NL_keywords_very_weak": ["combine data", "transform data", "data manipulation", "clean data", "reshape data", "data cleaning", "restructure data", "data alignment", "data conversion", "data modification", "data adjustment", "data transformation", "data normalization", "data preprocessing", "data transformation", "merge", "join", "concatenate", "groupby", "pivot", "filter", "map", "df.apply", "replace", "pd.melt", "pd.pivot_table", "pd.concat", "pd.merge", "pd.join", "pd.groupby", "pd.fillna", "pd.dropna", "pd.apply", "pd.map", "pd.replace", "pd.to_datetime", "pd.to_numeric", "astype", "pandas.get_dummies", "cut", "qcut", "DataFrameMapper", "ColumnTransformer", "normalize", "standardize", "MinMaxScaler", "LabelEncoder", "merge", "join", "concatenate", "groupby", "pivot", "filter", "map", "df.apply", "replace", "pd.melt", "pd.pivot_table", "pd.concat", "pd.merge", "pd.join", "pd.groupby", "pd.fillna", "pd.dropna", "pd.apply", "pd.map", "pd.replace", "pd.to_datetime", "pd.to_numeric", "astype", "pandas.get_dummies", "cut", "qcut", "DataFrameMapper", "ColumnTransformer", "normalize", "standardize", "MinMaxScaler", "LabelEncoder", "merge", "join", "concatenate", "groupby", "pivot", "filter", "map", "df.apply", "replace", "pd.melt", "pd.pivot_table", "pd.concat", "pd.merge", "pd.join", "pd.groupby", "pd.fillna", "pd.dropna", "pd.apply", "pd.map", "pd.replace", "pd.to_datetime", "pd.to_numeric", "astype", "pandas.get_dummies", "cut", "qcut", "DataFrameMapper", "ColumnTransformer", "normalize", "standardize", "MinMaxScaler", "LabelEncoder", "merge", "join", "concatenate", "groupby", "pivot", "filter", "map", "df.apply", "replace", "pd.melt", "pd.pivot_table", "pd.concat", "pd.merge", "pd.join", "pd.groupby", "pd.fillna", "pd.dropna", "pd.apply", "pd.map", "pd.replace", "pd.to_datetime", "pd.to_numeric", "astype", "pandas.get_dummies", "cut", "qcut", "DataFrameMapper", "ColumnTransformer", "normalize", "standardize", "MinMaxScaler", "LabelEncoder", "merge", "join", "concatenate", "groupby", "pivot", "filter", "map", "df.apply", "replace", "pd.melt", "pd.pivot_table", "pd.concat", "pd.merge", "pd.join", "pd.groupby", "pd.fillna", "pd.dropna", "pd.apply", "pd.map", "pd.replace", "pd.to_datetime", "pd.to_numeric", "astype", "pandas.get_dummies", "cut", "qcut", "DataFrameMapper", "ColumnTransformer", "normalize", "standardize", "MinMaxScaler", "LabelEncoder"],  
        },
        "Model_Train": {
            "PL_keywords": ["model.fit", "train_test_split", "cross_validate", "Sequential(", "estimator.fit", "compile", ".fit("],
            "NL_keywords": ["model training", "training model", "model fitting", "learning algorithm", "supervised training", "model architecture"],
            "PL_keywords_weak": ["model", "LinearRegression", "RandomForestClassifier", "KMeans", ".train(", "model_selection", "pipeline.Pipeline", "models.compile", "training_data"],
            "NL_keywords_weak": ["training algorithms", "learning models", "model development", "building model", "model creation", "learning from data", "training process", "model setup", "learning process"],
            #"PL_keywords_very_weak": ["fit", "train", "model", "regressor", "classifier", "predict", "sklearn.model_selection", "train_test_split", "xgboost.train", "model.fit", "train_test_split", "cross_validate", "Sequential(", "estimator.fit", "compile", "LinearRegression", "RandomForestClassifier", "KMeans", ".train(", "fit_transform", "model_selection", "pipeline.Pipeline", "models.compile", "training_data", "fit", "train", "model", "regressor", "classifier", "predict", ".fit(", "sklearn.model_selection", "train_test_split", "xgboost.train", "model.fit", "train_test_split", "cross_validate", "Sequential(", "estimator.fit", "compile", ".fit(", "LinearRegression", "RandomForestClassifier", "KMeans", ".train(", "fit_transform", "model_selection", "pipeline.Pipeline", "models.compile", "training_data", "fit", "train", "model", "regressor", "classifier", "predict", ".fit(", "sklearn.model_selection", "train_test_split", "xgboost.train", "model.fit", "train_test_split", "cross_validate", "Sequential(", "estimator.fit", "compile", ".fit(", "LinearRegression", "RandomForestClassifier", "KMeans", ".train(", "fit_transform", "model_selection", "pipeline.Pipeline", "models.compile", "training_data", "fit", "train", "model", "regressor", "classifier", "predict", ".fit(", "sklearn.model_selection", "train_test_split", "xgboost.train", "model.fit", "train_test_split", "cross_validate", "Sequential(", "estimator.fit", "compile", ".fit(", "LinearRegression", "RandomForestClassifier", "KMeans", ".train(", "fit_transform", "model_selection", "pipeline.Pipeline", "models.compile", "training_data", "fit", "train", "model", "regressor", "classifier", "predict", ".fit(", "sklearn.model_selection", "train_test_split", "xgboost.train", "model.fit", "train_test_split", "cross_validate", "Sequential(", "estimator.fit", "compile", ".fit(", "LinearRegression", "RandomForestClassifier", "KMeans", ".train(", "fit_transform", "model_selection", "pipeline.Pipeline", "models.compile", "training_data", "fit", "train", "model", "regressor", "classifier", "predict", ".fit(", "sklearn.model_selection", "train_test_split", "xgboost.train", "model.fit", "train_test_split", "cross_validate", "Sequential(", "estimator.fit", "compile", ".fit(", "LinearRegression", "RandomForestClassifier", "KMeans", ".train(", "fit_transform", "model_selection", "pipeline.Pipeline", "models.compile", "training_data", "fit", "train", "model", "regressor", "classifier", "predict", ".fit(", "sklearn.model_selection", "train_test_split", "xgboost.train", "model.fit", "train_test_split", "cross_validate", "Sequential(", "estimator.fit", "compile", ".fit(", "LinearRegression", "RandomForestClassifier", "KMeans", ".train(", "fit_transform", "model_selection", "pipeline.Pipeline", "models.compile", "training_data", "fit", "train", "model", "regressor", "classifier", "predict", ".fit(", "sklearn.model_selection", "train_test_split", "xgboost.train", "model.fit", "train_test_split", "cross_validate", "Sequential(", "estimator.fit", "compile", ".fit(", "LinearRegression", "RandomForestClassifier", "KMeans", ".train(", "fit_transform", "model_selection", "pipeline.Pipeline", "models.compile", "training_data", "fit", "train", "model", "regressor", "classifier", "predict", ".fit(", "sklearn.model_selection", "train_test_split", "xgboost.train", "model.fit", "train_test_split", "cross_validate", "Sequential(", "estimator.fit", "compile", ".fit(", "LinearRegression", "RandomForestClassifier", "KMeans", ".train(", "fit_transform", "model_selection", "pipeline.Pipeline", "models.compile", "training_data", "fit", "train", "model", "regressor", "classifier", "predict", ".fit(", "sklearn.model_selection", "train_test_split", "xgboost.train", "model.fit", "train_test_split", "cross_validate", "Sequential(", "estimator.fit", "compile", ".fit(", "LinearRegression", "RandomForestClassifier", "KMeans", ".train(", "fit_transform", "model_selection", "pipeline.Pipeline", "models.compile", "training_data", "fit", "train", "model", "regressor", "classifier", "predict", ".fit(", "sklearn.model_selection", "train_test_split", "xgboost.train", "model.fit", "train_test_split", "cross_validate", "Sequential(", "estimator.fit", "compile"],
            #"NL_keywords_very_weak": ["train model", "fit model", "model training", "supervised learning", "training phase", "learning process", "train classifier", "regression training", "classification training", "model fitting", "model training", "training model", "model fitting", "learning algorithm", "supervised training", "model architecture", "fit", "train", "model", "regressor", "classifier", "predict", ".fit(", "sklearn.model_selection", "train_test_split", "xgboost.train", "model.fit", "train_test_split", "cross_validate", "Sequential(", "estimator.fit", "compile", ".fit(", "LinearRegression", "RandomForestClassifier", "KMeans", ".train(", "fit_transform", "model_selection", "pipeline.Pipeline", "models.compile", "training_data", "fit", "train", "model", "regressor", "classifier", "predict", ".fit(", "sklearn.model_selection", "train_test_split", "xgboost.train", "model.fit", "train_test_split", "cross_validate", "Sequential(", "estimator.fit", "compile", ".fit(", "LinearRegression", "RandomForestClassifier", "KMeans", ".train(", "fit_transform", "model_selection", "pipeline.Pipeline", "models.compile", "training_data", "fit", "train", "model", "regressor", "classifier", "predict", ".fit(", "sklearn.model_selection", "train_test_split", "xgboost.train", "model.fit", "train_test_split", "cross_validate", "Sequential(", "estimator.fit", "compile", ".fit(", "LinearRegression", "RandomForestClassifier", "KMeans", ".train(", "fit_transform", "model_selection", "pipeline.Pipeline", "models.compile", "training_data", "fit", "train", "model", "regressor", "classifier", "predict", ".fit(", "sklearn.model_selection", "train_test_split", "xgboost.train", "model.fit", "train_test_split", "cross_validate", "Sequential(", "estimator.fit", "compile", ".fit(", "LinearRegression", "RandomForestClassifier", "KMeans", ".train(", "fit_transform", "model_selection", "pipeline.Pipeline", "models.compile", "training_data", "fit", "train", "model", "regressor", "classifier", "predict", ".fit(", "sklearn.model_selection", "train_test_split", "xgboost.train", "model.fit", "train_test_split"],

        },
        "Model_Evaluation": {
            "PL_keywords": ["model.evaluate", "confusion_matrix", "accuracy_score", "precision_score", "recall_score", "f1_score", ".evluate(", "precision", "recall", "f1_score", "roc_auc_score", "classification_report", "cross_val_score"],
            "NL_keywords": ["evaluation model", "model evaluation", "model performance", "evaluation metrics", "accuracy measurement", "model testing", "validation results"],
            "PL_keywords_weak": ["model", ".score(", "mean_squared_error", "log_loss", ".predict(", "validation_split", "train_validation_test", "model_selection.cross_val_predict", "evaluate_model", "testing_model"],
            "NL_keywords_weak": ["evaluating accuracy", "performance analysis", "results analysis", "accuracy testing", "validation of model", "test performance", "assessment of model", "model quality", "evaluation process"],
            #"PL_keywords_very_weak": ["model", "score", "evaluate", "test", "validation", "accuracy", "precision", "recall", "f1_score", "confusion_matrix", ".predict(", "model.evaluate", "confusion_matrix", "accuracy_score", "precision_score", "recall_score", "f1_score", ".evluate(", "precision", "recall", "f1_score", "roc_auc_score", "classification_report", "cross_val_score", "score", "evaluate", "test", "validation", "accuracy", "precision", "recall", "f1_score", "confusion_matrix", ".predict(", "model.evaluate", "confusion_matrix", "accuracy_score", "precision_score", "recall_score", "f1_score", ".evluate(", "precision", "recall", "f1_score", "roc_auc_score", "classification_report", "cross_val_score", "score", "evaluate", "test", "validation", "accuracy", "precision", "recall", "f1_score", "confusion_matrix", ".predict(", "model.evaluate", "confusion_matrix", "accuracy_score", "precision_score", "recall_score", "f1_score", ".evluate(", "precision", "recall", "f1_score", "roc_auc_score", "classification_report", "cross_val_score", "score", "evaluate", "test", "validation", "accuracy", "precision", "recall", "f1_score", "confusion_matrix", ".predict(", "model.evaluate", "confusion_matrix", "accuracy_score", "precision_score", "recall_score", "f1_score", ".evluate(", "precision", "recall", "f1_score", "roc_auc_score", "classification_report", "cross_val_score", "score", "evaluate", "test", "validation", "accuracy", "precision", "recall", "f1_score", "confusion_matrix", ".predict(", "model.evaluate", "confusion_matrix", "accuracy_score", "precision_score", "recall_score", "f1_score", ".evluate(", "precision", "recall", "f1_score", "roc_auc_score", "classification_report", "cross_val_score", "score", "evaluate", "test", "validation", "accuracy", "precision", "recall", "f1_score", "confusion_matrix", ".predict(", "model.evaluate", "confusion_matrix", "accuracy_score", "precision_score", "recall_score", "f1_score", ".evluate(", "precision", "recall", "f1_score", "roc_auc_score", "classification_report", "cross_val_score", "score", "evaluate", "test", "validation", "accuracy", "precision", "recall", "f1_score", "confusion_matrix", ".predict(", "model.evaluate", "confusion_matrix", "accuracy_score", "precision_score", "recall_score", "f1_score", ".evluate(", "precision", "recall", "f1_score", "roc_auc_score", "classification_report", "cross_val_score", "score", "evaluate", "test", "validation", "accuracy", "precision", "recall", "f1_score", "confusion_matrix", ".predict(", "model.evaluate", "confusion_matrix", "accuracy_score", "precision_score", "recall_score", "f1_score", ".evluate(", "precision", "recall", "f1_score", "roc_auc_score", "classification_report", "cross_val_score", "score", "evaluate", "test", "validation", "accuracy", "precision", "recall", "f1_score", "confusion_matrix", ".predict(", "model.evaluate", "confusion_matrix", "accuracy_score", "precision_score", "recall_score", "f1_score", ".evluate(", "precision", "recall", "f1_score", "roc_auc_score", "classification_report", "cross_val_score", "score", "evaluate", "test", "validation", "accuracy", "precision", "recall", "f1_score", "confusion_matrix", ".predict(", "model.evaluate", "confusion_matrix", "accuracy_score", "precision_score", "recall_score", "f1_score", ".evluate(", "precision", "recall", "f1_score", "roc_auc_score", "classification_report", "cross_val_score", "score", "evaluate", "test", "validation", "accuracy", "precision", "recall", "f1_score", "confusion_matrix", ".predict(", "model.evaluate", "confusion_matrix", "accuracy_score", "precision_score", "recall_score", "f1_score", ".evluate(", "precision", "recall", "f1_score", "roc_auc_score", "classification_report", "cross_val_score", "score", "evaluate", "test", "validation", "accuracy", "precision", "recall", "f1_score", "confusion_matrix", ".predict(", "model.evaluate", "confusion_matrix", "accuracy"],
            #"NL_keywords_very_weak": ["model evaluation", "test model", "model performance", "accuracy of the model", "evaluate results", "performance metrics", "validation results", "testing phase", "model assessment", "model quality", "evaluation model", "model evaluation", "model performance", "evaluation metrics", "accuracy measurement", "model testing", "validation results", "evaluate", "test", "validation", "accuracy", "precision", "recall", "f1_score", "confusion_matrix", ".predict(", "model.evaluate", "confusion_matrix", "accuracy_score", "precision_score", "recall_score", "f1_score", ".evluate(", "precision", "recall", "f1_score", "roc_auc_score", "classification_report", "cross_val_score", "evaluate", "test", "validation", "accuracy", "precision", "recall", "f1_score", "confusion_matrix", ".predict(", "model.evaluate", "confusion_matrix", "accuracy_score", "precision_score", "recall_score", "f1_score", ".evluate(", "precision", "recall", "f1_score", "roc_auc_score", "classification_report", "cross_val_score", "evaluate", "test", "validation", "accuracy", "precision", "recall", "f1_score", "confusion_matrix", ".predict(", "model.evaluate", "confusion_matrix", "accuracy_score", "precision_score", "recall_score", "f1_score", ".evluate(", "precision", "recall", "f1_score", "roc_auc_score", "classification_report", "cross_val_score", "evaluate", "test", "validation", "accuracy", "precision", "recall", "f1_score", "confusion_matrix", ".predict(", "model.evaluate", "confusion_matrix", "accuracy_score", "precision_score", "recall_score", "f1_score", ".evluate(", "precision", "recall", "f1_score", "roc_auc_score", "classification_report", "cross_val_score", "evaluate", "test", "validation", "accuracy", "precision", "recall", "f1_score", "confusion_matrix", ".predict(", "model.evaluate", "confusion_matrix", "accuracy_score", "precision_score", "recall_score", "f1_score", ".evluate(", "precision", "recall", "f1_score", "roc_auc_score", "classification_report", "cross_val_score", "evaluate", "test", "validation", "accuracy", "precision", "recall", "f1_score", "confusion_matrix"],
        },
        "Model_Interpretation": {
            "PL_keywords": ["feature_importances_", "shap_values", "plot_tree", "partial_dependence_plot", "eli5.show_weights", "LIME", "SHAP", "PermutationImportance", "plot_importance", "plot_partial_dependence", "plot_pdp", "plot_shap", "plot_lime", "plot_permutation_importance"],
            "NL_keywords": ["model interpretation", "interpretation model", "importance feature", "feature importance", "model insights", "interpretation techniques", "model explanation", "decision explanations"],
            "PL_keywords_weak": ["model", "explain_model", "model_explainability", "FeatureImportance", "decision_path", "visualize_model", "interpret_model", "explain_predictions", "feature_contributions", "impact"],
            "NL_keywords_weak": ["understanding model", "model analysis", "explanation of predictions", "feature analysis", "model transparency", "interpretability techniques", "underlying patterns", "model rationale", "decision-making insight"],
            #"PL_keywords_very_weak": ["model", "feature_importance_", "coef_", "decision_function", "partial_dependence", "shap_values", "eli5.show_weights", "interpret_model", "LIME", "SHAP", "PermutationImportance", "plot_importance", "plot_partial_dependence", "plot_pdp", "plot_shap", "plot_lime", "plot_permutation_importance", "feature_importance_", "coef_", "decision_function", "partial_dependence", "shap_values", "eli5.show_weights", "interpret_model", "LIME", "SHAP", "PermutationImportance", "plot_importance", "plot_partial_dependence", "plot_pdp", "plot_shap", "plot_lime", "plot_permutation_importance", "feature_importance_", "coef_", "decision_function", "partial_dependence", "shap_values", "eli5.show_weights", "interpret_model", "LIME", "SHAP", "PermutationImportance", "plot_importance", "plot_partial_dependence", "plot_pdp", "plot_shap", "plot_lime", "plot_permutation_importance", "feature_importance_", "coef_", "decision_function", "partial_dependence", "shap_values", "eli5.show_weights", "interpret_model", "LIME", "SHAP", "PermutationImportance", "plot_importance", "plot_partial_dependence", "plot_pdp", "plot_shap", "plot_lime", "plot_permutation_importance", "feature_importance_", "coef_", "decision_function", "partial_dependence", "shap_values", "eli5.show_weights", "interpret_model", "LIME", "SHAP", "PermutationImportance", "plot_importance", "plot_partial_dependence", "plot_pdp", "plot_shap", "plot_lime", "plot_permutation_importance", "feature_importance_", "coef_", "decision_function", "partial_dependence", "shap_values", "eli5.show_weights", "interpret_model", "LIME", "SHAP", "PermutationImportance", "plot_importance", "plot_partial_dependence", "plot_pdp", "plot_shap", "plot_lime", "plot_permutation_importance", "feature_importance_", "coef_", "decision_function", "partial_dependence", "shap_values", "eli5.show_weights", "interpret_model", "LIME", "SHAP", "PermutationImportance", "plot_importance", "plot_partial_dependence", "plot_pdp", "plot_shap", "plot_lime", "plot_permutation_importance", "feature_importance_", "coef_", "decision_function", "partial_dependence", "shap_values", "eli5.show_weights", "interpret_model", "LIME", "SHAP", "PermutationImportance", "plot_importance", "plot_partial_dependence", "plot_pdp", "plot_shap", "plot_lime", "plot_permutation_importance", "feature_importance_", "coef_", "decision_function", "partial_dependence", "shap_values", "eli5.show_weights", "interpret_model", "LIME", "SHAP", "PermutationImportance", "plot_importance", "plot_partial_dependence", "plot_pdp", "plot_shap", "plot_lime", "plot_permutation_importance", "feature_importance_", "coef_", "decision_function", "partial_dependence", "shap_values", "eli5.show_weights", "interpret_model", "LIME", "SHAP", "PermutationImportance", "plot_importance", "plot_partial_dependence", "plot_pdp", "plot_shap", "plot_lime", "plot_permutation_importance", "feature_importance_", "coef_", "decision_function", "partial_dependence", "shap_values", "eli5.show_weights", "interpret_model", "LIME", "SHAP", "PermutationImportance", "plot_importance", "plot_partial_dependence", "plot_pdp", "plot_shap", "plot_lime", "plot_permutation_importance", "feature_importance_", "coef_", "decision_function", "partial_dependence", "shap_values", "eli5.show_weights", "interpret_model", "LIME", "SHAP", "PermutationImportance", "plot_importance", "plot_partial_dependence", "plot_pdp", "plot_shap", "plot_lime", "plot_permutation_importance", "feature_importance_", "coef_", "decision_function", "partial_dependence", "shap_values", "eli5.show_weights", "interpret_model", "LIME", "SHAP", "PermutationImportance", "plot_importance", "plot_partial_dependence", "plot_pdp", "plot_shap", "plot_lime", "plot_permutation_importance", "feature_importance_", "coef_"],
            #"NL_keywords_very_weak": ["interpret model", "model insights", "feature importance", "understand model", "model explanation", "explain predictions", "interpretation techniques", "model reasoning", "decision-making insight", "model interpretation", "interpretation model", "importance feature", "feature importance", "model insights", "interpretation techniques", "model explanation", "decision explanations", "feature_importances_", "shap_values", "plot_tree", "partial_dependence_plot", "eli5.show_weights", "LIME", "SHAP", "PermutationImportance", "plot_importance", "plot_partial_dependence", "plot_pdp", "plot_shap", "plot_lime", "plot_permutation_importance", "feature_importances_", "shap_values", "plot_tree", "partial_dependence_plot", "eli5.show_weights", "LIME", "SHAP", "PermutationImportance", "plot_importance", "plot_partial_dependence", "plot_pdp", "plot_shap", "plot_lime", "plot_permutation_importance", "feature_importances_", "shap_values", "plot_tree", "partial_dependence_plot", "eli5.show_weights", "LIME", "SHAP", "PermutationImportance", "plot_importance", "plot_partial_dependence", "plot_pdp", "plot_shap", "plot_lime", "plot_permutation_importance", "feature_importances_", "shap_values", "plot_tree", "partial_dependence_plot", "eli5.show_weights", "LIME", "SHAP", "PermutationImportance", "plot_importance", "plot_partial_dependence", "plot_pdp", "plot_shap", "plot_lime", "plot_permutation_importance", "feature_importances_", "shap_values", "plot_tree", "partial_dependence_plot", "eli5.show_weights", "LIME", "SHAP", "PermutationImportance", "plot_importance", "plot_partial_dependence", "plot_pdp", "plot_shap", "plot_lime", "plot_permutation_importance", "feature_importances_", "shap_values", "plot_tree", "partial_dependence_plot", "eli5.show_weights", "LIME", "SHAP", "PermutationImportance", "plot_importance", "plot_partial_dependence", "plot_pdp", "plot_shap", "plot_lime", "plot_permutation_importance", "feature_importances_", "shap_values", "plot_tree", "partial_dependence_plot"],
        },
        "Hyperparameter_Tuning": {
            "PL_keywords": ["GridSearchCV", "RandomizedSearchCV", "param_grid", "n_iter", "BayesianOptimization", "Optuna", "early_stopping", "hyperopt", "tune_grid", "tune_random", "tune_bayesian", "tune_optuna"],
            "NL_keywords": ["hyperparameter tuning", "tune hyperparameters", "parameter optimization", "model tuning", "search strategy", "optimization techniques", "tuning parameters"],
            "PL_keywords_weak": ["param", "tune_model", "model_optimization", "search_parameters", "hyperparameter", "Hyperparameters", "parameter_search", "optimize_model", "adjust_parameters", "tuning_strategy", "optimization_process"],
            "NL_keywords_weak": ["adjusting model", "optimizing performance", "parameter adjustments", "fine-tuning models", "optimization of parameters", "model optimization", "tuning process", "optimization strategies", "parameter selection"],
            #"PL_keywords_very_weak": ["GridSearchCV", "RandomizedSearchCV", "cross_val_score", "param_grid", "hyperopt.fmin", "BayesianOptimization", "model_selection", "tune.run", "tune_grid", "tune_random", "tune_bayesian", "tune_optuna", "GridSearchCV", "RandomizedSearchCV", "cross_val_score", "param_grid", "hyperopt.fmin", "BayesianOptimization", "model_selection", "tune.run", "tune_grid", "tune_random", "tune_bayesian", "tune_optuna", "GridSearchCV", "RandomizedSearchCV", "cross_val_score", "param_grid", "hyperopt.fmin", "BayesianOptimization", "model_selection", "tune.run", "tune_grid", "tune_random", "tune_bayesian", "tune_optuna", "GridSearchCV", "RandomizedSearchCV", "cross_val_score", "param_grid", "hyperopt.fmin", "BayesianOptimization", "model_selection", "tune.run", "tune_grid", "tune_random", "tune_bayesian", "tune_optuna", "GridSearchCV", "RandomizedSearchCV", "cross_val_score", "param_grid", "hyperopt.fmin", "BayesianOptimization", "model_selection", "tune.run", "tune_grid", "tune_random", "tune_bayesian", "tune_optuna", "GridSearchCV", "RandomizedSearchCV", "cross_val_score", "param_grid", "hyperopt.fmin", "BayesianOptimization", "model_selection", "tune.run", "tune_grid", "tune_random", "tune_bayesian", "tune_optuna", "GridSearchCV", "RandomizedSearchCV", "cross_val_score", "param_grid", "hyperopt.fmin", "BayesianOptimization", "model_selection", "tune.run", "tune_grid", "tune_random", "tune_bayesian", "tune_optuna", "GridSearchCV", "RandomizedSearchCV", "cross_val_score", "param_grid", "hyperopt.fmin", "BayesianOptimization", "model_selection", "tune.run", "tune_grid", "tune_random", "tune_bayesian", "tune_optuna", "GridSearchCV", "RandomizedSearchCV", "cross_val_score", "param_grid", "hyperopt.fmin"],
            #"NL_keywords_very_weak": ["tune parameters", "hyperparameter optimization", "parameter search", "optimize model", "tuning strategy", "optimize performance", "search strategy", "parameter optimization", "model tuning", "search strategy", "optimization techniques", "tuning parameters", "tune parameters", "hyperparameter optimization", "parameter search", "optimize model", "tuning strategy", "optimize performance", "search strategy", "parameter optimization", "model tuning", "search strategy", "optimization techniques", "tuning parameters", "tune parameters", "hyperparameter optimization", "parameter search", "optimize model", "tuning strategy", "optimize performance", "search strategy", "parameter optimization", "model tuning", "search strategy", "optimization techniques", "tuning parameters", "tune parameters", "hyperparameter optimization", "parameter search", "optimize model", "tuning strategy", "optimize performance", "search strategy", "parameter optimization", "model tuning", "search strategy", "optimization techniques", "tuning parameters", "tune parameters", "hyperparameter optimization", "parameter search", "optimize model", "tuning strategy", "optimize performance", "search strategy", "parameter optimization", "model tuning", "search strategy", "optimization techniques", "tuning parameters", "tune parameters", "hyperparameter optimization", "parameter search", "optimize model", "tuning strategy", "optimize performance", "search strategy", "parameter optimization", "model tuning", "search strategy", "optimization techniques", "tuning parameters", "tune parameters", "hyperparameter optimization", "parameter search", "optimize model", "tuning strategy", "optimize performance", "search strategy", "parameter optimization", "model tuning", "search strategy", "optimization techniques", "tuning parameters", "tune parameters", "hyperparameter optimization", "parameter search", "optimize model", "tuning strategy", "optimize performance", "search strategy", "parameter optimization", "model tuning", "search strategy", "optimization techniques", "tuning parameters", "tune parameters", "hyperparameter optimization", "parameter search", "optimize model", "tuning strategy", "optimize performance", "search strategy", "parameter optimization", "model tuning", "search strategy", "optimization techniques", "tuning parameters", "tune parameters", "hyperparameter optimization", "parameter search", "optimize model", "tuning strategy", "optimize performance", "search strategy", "parameter optimization", "model tuning", "search strategy", "optimization techniques", "tuning parameters", "tune parameters", "hyperparameter optimization", "parameter search", "optimize model", "tuning strategy", "optimize performance", "search strategy", "parameter optimization", "model tuning", "search strategy", "optimization techniques", "tuning parameters", "tune parameters", "hyperparameter optimization", "parameter search", "optimize model", "tuning strategy", "optimize performance", "search strategy", "parameter optimization", "model tuning", "search strategy", "optimization techniques", "tuning parameters", "tune parameters", "hyperparameter optimization", "parameter search", "optimize model", "tuning strategy", "optimize performance", "search strategy", "parameter optimization", "model tuning", "search strategy", "optimization techniques", "tuning parameters"],
        },
        "Visualization": {
            "PL_keywords": ["plt.plot", "df.hist", "sns.scatterplot", "sns.heatmap", "plt.boxplot", "fig, ax = plt.subplots", "plt.bar", ".hist(", ".plot(", ".scatter(", ".boxplot(", ".heatmap(", ".bar("],
            "NL_keywords": ["plot data", "data plotting", "visualization tools", "charts and graphs", "visual insights", "plotting data", "graphical representation", "data visualization"],
            "PL_keywords_weak": ["sns.", "seaborn.", "plotly.express", "matplotlib", "seaborn.factorplot", "bokeh.plotting", "ggplot", "pyplot", "visualize", "chart", "graph"],
            "NL_keywords_weak": ["creating plots", "drawing graphs", "charting data", "visual representations", "data charts", "graphical plots", "visual analysis tools", "data graphics", "plot creation"],
            #"PL_keywords_very_weak": ["plot", "show", "figure", "hist", "scatter", "bar", "lineplot", "plt.xlabel", "sns.heatmap", "plotly.graph_objs", "plotly.offline", "plotly.subplots", "plotly.express", "plotly.subplots", "plotly.offline", "plotly.graph_objs", "plotly.express", "plotly.offline", "plotly.subplots", "plotly.express", "plotly.offline", "plotly.graph_objs", "plotly.express", "plotly.offline", "plotly.subplots", "plotly.express", "plotly.offline", "plotly.graph_objs", "plotly.express", "plotly.offline", "plotly.subplots", "plotly.express", "plotly.offline", "plotly.graph_objs", "plotly.express", "plotly.offline", "plotly.subplots", "plotly.express", "plotly.offline", "plotly.graph_objs", "plotly.express", "plotly.offline", "plotly.subplots", "plotly.express", "plotly.offline", "plotly.graph_objs", "plotly.express", "plotly.offline", "plotly.subplots", "plotly.express", "plotly.offline", "plotly.graph_objs", "plotly.express", "plotly.offline", "plotly.subplots", "plotly.express", "plotly.offline", "plotly.graph_objs", "plotly.express", "plotly.offline", "plotly.subplots", "plotly.express", "plotly.offline", "plotly.graph_objs", "plotly.express", "plotly.offline", "plotly.subplots", "plotly.express", "plotly.offline", "plotly.graph_objs", "plotly.express", "plotly.offline", "plotly.subplots", "plotly.express", "plotly.offline", "plotly.graph_objs", "plotly.express", "plotly.offline", "plotly.subplots", "plotly.express", "plotly.offline", "plotly.graph_objs", "plotly.express", "plotly.offline", "plotly.subplots", "plotly.express", "plotly.offline", "plotly.graph_objs", "plotly.express", "plotly.offline", "plotly.subplots", "plotly.express", "plotly.offline", "plotly.graph_objs", "plotly.express", "plotly.offline", "plotly.subplots", "plotly.express", "plotly.offline", "plotly.graph_objs", "plotly.express", "plotly.offline", "plotly.subplots", "plotly.express", "plotly.offline", "plotly.graph_objs", "plotly.express", "plotly.offline", "plotly.subplots", "plotly.express", "plotly.offline", "plotly.graph_objs", "plotly.express", "plotly.offline", "plotly.subplots", "plotly.express", "plotly.offline", "plotly.graph_objs", "plotly.express", "plotly.offline", "plotly.subplots", "plotly.express", "plotly.offline", "plotly.graph_objs", "plotly.express", "plotly.offline", "plotly.subplots", "plotly.express", "plotly.offline", "plotly.graph_objs", "plotly.express", "plotly.offline", "plotly.subplots", "plotly.express", "plotly.offline", "plotly.graph_objs", "plotly.express", "plotly.offline", "plotly.subplots", "plotly.express", "plotly.offline", "plotly.graph_objs", "plotly.express", "plotly.offline", "plotly.subplots", "plotly.express", "plotly.offline", "plotly.graph_objs", "plotly.express", "plotly.offline", "plotly.subplots", "plotly.express", "plotly.offline", "plotly.graph_objs", "plotly.express", "plotly.offline", "plotly.subplots", "plotly.express", "plotly.offline", "plotly.graph_objs", "plotly.express", "plotly.offline", "plotly.subplots", "plotly.express", "plotly.offline", "plotly.graph_objs", "plotly.express", "plotly.offline", "plotly.subplots", "plotly.express", "plotly.offline", "plotly.graph_objs", "plotly.express", "plotly.offline", "plotly.subplots", "plotly.express", "plotly.offline", "plotly.graph_objs", "plotly.express", "plotly.offline", "plotly.subplots", "plotly.express"],
            #"NL_keywords_very_weak": ["plot data", "visualize", "chart", "graph", "data visualization", "show data", "illustrate data", "visualization techniques", "chart analysis", "graphical representation", "plot data", "visualize", "chart", "graph", "data visualization", "show data", "illustrate data", "visualization techniques", "chart analysis", "graphical representation", "plot data", "visualize", "chart", "graph", "data visualization", "show data", "illustrate data", "visualization techniques", "chart analysis", "graphical representation", "plot data", "visualize", "chart", "graph", "data visualization", "show data", "illustrate data", "visualization techniques", "chart analysis", "graphical representation", "plot data", "visualize", "chart", "graph", "data visualization", "show data", "illustrate data", "visualization techniques", "chart analysis", "graphical representation", "plot data", "visualize", "chart", "graph", "data visualization", "show data", "illustrate data", "visualization techniques", "chart analysis", "graphical representation", "plot data", "visualize", "chart", "graph", "data visualization", "show data", "illustrate data", "visualization techniques", "chart analysis", "graphical representation", "plot data", "visualize", "chart", "graph", "data visualization", "show data", "illustrate data", "visualization techniques", "chart analysis", "graphical representation", "plot data", "visualize", "chart", "graph", "data visualization", "show data", "illustrate data", "visualization techniques", "chart analysis", "graphical representation", "plot data", "visualize", "chart", "graph", "data visualization", "show data", "illustrate data", "visualization techniques", "chart analysis", "graphical representation", "plot data", "visualize", "chart", "graph", "data visualization", "show data", "illustrate data", "visualization techniques", "chart analysis", "graphical representation", "plot data", "visualize", "chart", "graph", "data visualization", "show data", "illustrate data", "visualization techniques", "chart analysis", "graphical representation", "plot data", "visualize", "chart", "graph", "data visualization", "show data", "illustrate data", "visualization techniques", "chart analysis", "graphical representation", "plot data", "visualize", "chart", "graph"],
        },
        "Debug": {
            "PL_keywords": ["assert", "try:", "except:", "pdb.set_trace", "logging.debug", "sys.exc_info"],
            "NL_keywords": ["debug", "error analysis", "troubleshooting steps", "debugging code", "error resolution", "problem solving", "code debugging", "error handling"],
            "PL_keywords_weak": ["debugger", "traceback", "exception", "error_log", "sys.tracebacklimit", "debug_mode", "stack_trace", "breakpoint", "error_report"],
            "NL_keywords_weak": ["finding errors", "code analysis", "fixing bugs", "problem identification", "error detection", "troubleshooting guide", "solving issues", "code inspection", "debugging steps"],
            #"PL_keywords_very_weak": ["debug", "error", "exception", "traceback", "breakpoint", "assert", "try:", "except:", "pdb.set_trace()", "logging.debug", "sys.exc_info", "debug", "error", "exception", "traceback", "breakpoint", "assert", "try:", "except:", "pdb.set_trace()", "logging.debug", "sys.exc_info", "debug", "error", "exception", "traceback", "breakpoint", "assert", "try:", "except:", "pdb.set_trace()", "logging.debug", "sys.exc_info", "debug", "error", "exception", "traceback", "breakpoint", "assert", "try:", "except:", "pdb.set_trace()", "logging.debug", "sys.exc_info", "debug", "error", "exception", "traceback", "breakpoint", "assert", "try:", "except:", "pdb.set_trace()", "logging.debug", "sys.exc_info", "debug", "error", "exception", "traceback", "breakpoint", "assert", "try:", "except:", "pdb.set_trace()", "logging.debug", "sys.exc_info", "debug", "error", "exception", "traceback", "breakpoint", "assert", "try:", "except:", "pdb.set_trace()", "logging.debug", "sys.exc_info", "debug", "error", "exception", "traceback", "breakpoint", "assert", "try:", "except:", "pdb.set_trace()", "logging.debug", "sys.exc_info", "debug", "error", "exception", "traceback", "breakpoint", "assert", "try:", "except:", "pdb.set_trace()", "logging.debug", "sys.exc_info", "debug", "error", "exception", "traceback", "breakpoint", "assert", "try:", "except:", "pdb.set_trace()", "logging.debug", "sys.exc_info", "debug", "error", "exception", "traceback", "breakpoint", "assert", "try:", "except:", "pdb.set_trace()", "logging.debug", "sys.exc_info", "debug", "error", "exception", "traceback", "breakpoint", "assert", "try:", "except:", "pdb.set_trace()", "logging.debug", "sys.exc_info", "debug", "error", "exception", "traceback", "breakpoint", "assert", "try:", "except:", "pdb.set_trace()", "logging.debug", "sys.exc_info", "debug", "error", "exception", "traceback", "break", "assert", "try:", "except:", "pdb.set_trace()", "logging.debug", "sys.exc_info", "debug", "error", "exception", "traceback", "breakpoint", "assert", "try:", "except:", "pdb.set_trace()", "logging.debug", "sys.exc_info", "debug", "error", "exception", "traceback", "breakpoint", "assert", "try:", "except:", "pdb.set_trace()", "logging.debug", "sys.exc_info", "debug", "error", "exception", "traceback", "breakpoint", "assert", "try:", "except:", "pdb.set_trace()", "logging.debug", "sys.exc_info", "debug", "error", "exception", "traceback", "breakpoint", "assert", "try:", "except:", "pdb.set_trace()", "logging.debug", "sys.exc_info", "debug", "error", "exception", "traceback", "breakpoint", "assert", "try:", "except:", "pdb.set_trace()", "logging.debug", "sys.exc_info", "debug", "error", "exception", "traceback", "breakpoint", "assert", "try:", "except:", "pdb.set_trace()", "logging.debug", "sys.exc_info", "debug", "error", "exception", "traceback", "breakpoint", "assert", "try:", "except:", "pdb.set_trace()", "logging.debug", "sys.exc_info", "debug", "error", "exception", "traceback", "breakpoint", "assert", "try:", "except:", "pdb.set_trace()", "logging.debug", "sys.exc_info", "debug", "error", "exception", "traceback", "breakpoint", "assert", "try:", "except:", "pdb.set_trace()", "logging.debug", "sys.exc_info", "debug", "error", "exception", "traceback", "breakpoint", "assert", "try:", "except:", "pdb.set_trace()", "logging.debug", "sys.exc_info", "debug", "error", "exception", "traceback", "breakpoint", "assert", "try:", "except:", "pdb.set_trace()", "logging.debug", "sys.exc_info", "debug", "error", "exception", "traceback", "breakpoint", "assert", "try:", "except:", "pdb.set_trace()", "logging.debug", "sys"],
            #"NL_keywords_very_weak": ["fix error", "debugging", "error handling", "troubleshoot", "resolve issues", "problem solving", "code correction", "error analysis", "error", "debug", "exception", "traceback", "breakpoint", "assert", "try:", "except:", "pdb.set_trace()", "logging.debug", "sys.exc_info", "error", "debug", "exception", "traceback", "breakpoint", "assert", "try:", "except:", "pdb.set_trace()", "logging.debug", "sys.exc_info", "error", "debug", "exception", "traceback", "breakpoint", "assert", "try:", "except:", "pdb.set_trace()", "logging.debug", "sys.exc_info", "error", "debug", "exception", "traceback", "breakpoint", "assert", "try:", "except:", "pdb.set_trace()", "logging.debug", "sys.exc_info", "error", "debug", "exception", "traceback", "breakpoint", "assert", "try:", "except:", "pdb.set_trace()", "logging.debug", "sys.exc_info", "error", "debug", "exception", "traceback", "breakpoint", "assert", "try:", "except:", "pdb.set_trace()", "logging.debug", "sys.exc_info", "error", "debug", "exception", "traceback", "breakpoint", "assert", "try:", "except:", "pdb.set_trace()", "logging.debug", "sys.exc_info", "error", "debug", "exception", "traceback", "breakpoint", "assert", "try:", "except:", "pdb.set_trace()", "logging.debug", "sys.exc_info", "error", "debug", "exception", "traceback", "breakpoint", "assert", "try:", "except:", "pdb.set_trace()", "logging.debug", "sys.exc_info", "error", "debug", "exception", "traceback", "breakpoint", "assert", "try:", "except:", "pdb.set_trace()", "logging.debug", "sys.exc_info", "error", "debug", "exception", "traceback", "breakpoint", "assert", "try:", "except:", "pdb.set_trace()", "logging.debug", "sys.exc_info", "error", "debug", "exception", "traceback", "breakpoint", "assert", "try:", "except:", "pdb.set_trace()", "logging.debug", "sys.exc_info", "error", "debug", "exception", "traceback", "breakpoint", "assert", "try:"],
        },
        "Data_Export": {
            "PL_keywords": [".to_csv(", ".to_excel(", ".to_sql(", ".to_json(", ".to_parquet(", "pickle.dump", "np.save"],
            "NL_keywords": ["data export", "export data", "data saving", "exporting results", "writing data", "output files", "saving tables"],
            "PL_keywords_weak": ["write", "to_hdf", "to_dict", "savefig", "to_markdown", "to_clipboard", "export", "save_data", "store"],
            "NL_keywords_weak": ["saving results", "data storage", "writing output", "export process", "result storage", "output generation", "saving process", "data archiving", "record saving"],
            #"PL_keywords_very_weak": ["to_csv", "to_excel", "save", "write", "output", "export", "df.to_json", "pickle.dump", "np.save", "torch.save", "to_csv", "to_excel", "save", "write", "output", "export", "df.to_json", "pickle.dump", "np.save", "torch.save", "to_csv", "to_excel", "save", "write", "output", "export", "df.to_json", "pickle.dump", "np.save", "torch.save", "to_csv", "to_excel", "save", "write", "output", "export", "df.to_json", "pickle.dump", "np.save", "torch.save", "to_csv", "to_excel", "save", "write", "output", "export", "df.to_json", "pickle.dump", "np.save", "torch.save", "to_csv", "to_excel", "save", "write", "output", "export", "df.to_json", "pickle.dump", "np.save", "torch.save", "to_csv", "to_excel", "save", "write", "output", "export", "df.to_json", "pickle.dump", "np.save", "torch.save", "to_csv", "to_excel", "save", "write", "output", "export", "df.to_json", "pickle.dump", "np.save", "torch.save", "to_csv", "to_excel", "save", "write", "output", "export", "df.to_json", "pickle.dump", "np.save", "torch.save", "to_csv", "to_excel", "save", "write", "output", "export", "df.to_json", "pickle.dump", "np.save", "torch.save", "to_csv", "to_excel", "save", "write", "output", "export", "df.to_json", "pickle.dump", "np.save", "torch.save", "to_csv", "to_excel", "save", "write", "output", "export", "df.to_json", "pickle.dump", "np.save", "torch.save", "to_csv", "to_excel", "save", "write", "output", "export", "df.to_json", "pickle.dump", "np.save", "torch.save", "to_csv", "to_excel", "save", "write", "output", "export", "df.to_json", "pickle.dump", "np.save", "torch.save", "to_csv", "to_excel", "save", "write", "output", "export", "df.to_json", "pickle.dump", "np.save", "torch.save", "to_csv", "to_excel", "save", "write", "output", "export", "df.to_json", "pickle.dump", "np.save", "torch.save", "to_csv", "to_excel", "save", "write", "output", "export", "df.to_json", "pickle.dump", "np.save", "torch.save", "to_csv", "to_excel", "save", "write", "output", "export", "df.to_json", "pickle.dump", "np.save", "torch.save", "to_csv", "to_excel", "save", "write", "output", "export", "df.to_json", "pickle.dump", "np.save", "torch.save", "to_csv", "to_excel", "save", "write", "output", "export", "df.to_json", "pickle.dump", "np.save", "torch.save", "to_csv", "to_excel", "save", "write", "output", "export", "df.to_json", "pickle.dump", "np.save", "torch.save", "to_csv", "to_excel", "save", "write", "output", "export", "df.to_json", "pickle.dump", "np.save", "torch.save", "to_csv", "to_excel", "save", "write", "output", "export", "df.to_json", "pickle.dump", "np.save", "torch.save", "to_csv", "to_excel", "save", "write", "output", "export", "df.to_json", "pickle.dump", "np.save", "torch.save", "to_csv", "to_excel", "save", "write", "output", "export", "df.to_json", "pickle.dump", "np.save", "torch.save", "to_csv", "to_excel", "save", "write", "output", "export", "df.to_json", "pickle.dump", "np.save", "torch.save", "to_csv", "to_excel", "save", "write", "output", "export", "df.to_json", "pickle.dump", "np.save", "torch.save", "to_csv", "to_excel", "save", "write", "output", "export", "df.to_json", "pickle.dump", "np.save", "torch.save"],
            #"NL_keywords_very_weak": ["save data", "export data", "write file", "output data", "data output", "store results", "save results", "data preservation", "export analysis", "write data"],
        },
        "Other": {
            "PL_keywords": ["empty"],
            "NL_keywords": ["empty"],
            "PL_keywords_weak": ["empty"],
            "NL_keywords_weak": ["empty"],
            #"PL_keywords_very_weak": ["empty"],
            #"NL_keywords_very_weak": ["empty"],
        }
    }

    return hardcoded_keywords


def hardcoded_sklearn_keywords():
    """
    Function to hardcode a dictionary with sci-kit learn classes and functions based on their functionality.

    :return (dict): dictionary with sci-kit learn classes and functions based on their functionality
    """

    sklearn_keywords = {
        'Environment': {'sklearn': []},
        'Data_Extraction': {'sklearn': []},
        'Exploratory_Data_Analysis': {'sklearn': []},
        'Data_Transform': {'sklearn': []},
        'Model_Train': {'sklearn': []},
        'Model_Evaluation': {'sklearn': []},
        'Model_Interpretation': {'sklearn': []},
        'Hyperparameter_Tuning': {'sklearn': []},
        'Visualization': {'sklearn': []},
        'Debug': {'sklearn': []},
        'Data_Export': {'sklearn': []},
        'Other': {'sklearn': []},
    }

    # Environment-related classes and functions
    sklearn_keywords['Environment']['sklearn'].extend([
        'check_build', 'basic_check_build', 'compile_test_program', 'config_context', 
        'get_config', 'set_config', 'ConvergenceWarning', 'DataConversionWarning', 
        'DataDimensionalityWarning', 'EfficiencyWarning', 'FitFailedWarning', 'ValidationWarning'
   
    ])

    # Data Extraction-related classes and functions
    sklearn_keywords['Data_Extraction']['sklearn'].extend([
        'fetch_openml', 'load_iris', 'load_wine', 'load_digits', 'load_diabetes', 'load_linnerud',
        'load_boston', 'load_breast_cancer', 'fetch_20newsgroups', 'fetch_olivetti_faces',
        'fetch_lfw_people', 'fetch_lfw_pairs', 'fetch_rcv1', 'fetch_kddcup99', 'fetch_california_housing',
        'make_classification', 'make_regression', 'make_blobs', 'make_moons', 'make_circles',
        'fetch_20newsgroups_vectorized_fxt', 'fetch_covtype_fxt'
    ])

    # Exploratory Data Analysis-related classes and functions
    sklearn_keywords['Exploratory_Data_Analysis']['sklearn'].extend([
        'is_classifier', 'is_regressor', 'is_outlier_detector', 'BaseEstimator', 'TransformerMixin',
        'BiclusterMixin', 'ClassifierMixin', 'ClusterMixin', 'RegressorMixin', 'OutlierMixin',
        'check_array', 'check_X_y', 'check_is_fitted', 'check_random_state', 'check_symmetric',
        'check_consistent_length', 'check_scalar', 'check_pandas_support', 'check_memory',
    ])

    # Data Transformation-related classes and functions
    sklearn_keywords['Data_Transform']['sklearn'].extend([
        'StandardScaler', 'MinMaxScaler', 'OneHotEncoder', 'LabelEncoder', 'Binarizer',
        'PolynomialFeatures', 'CountVectorizer', 'TfidfVectorizer', 'HashingVectorizer',
        'Normalizer', 'FunctionTransformer', 'PowerTransformer', 'QuantileTransformer',
        'RobustScaler', 'OrdinalEncoder', 'ColumnTransformer', 'FeatureUnion',
        'PCA', 'NMF', 'FastICA', 'TruncatedSVD', 'RandomTreesEmbedding', 'RBFSampler', 
        'AdditiveChi2Sampler'
    ])

    # Model Training-related classes and functions
    sklearn_keywords['Model_Train']['sklearn'].extend([
        'LinearRegression', 'LogisticRegression', 'Ridge', 'Lasso', 'ElasticNet',
        'SGDClassifier', 'SGDRegressor', 'RandomForestClassifier', 'RandomForestRegressor',
        'GradientBoostingClassifier', 'GradientBoostingRegressor', 'DecisionTreeClassifier',
        'DecisionTreeRegressor', 'KNeighborsClassifier', 'KNeighborsRegressor', 'SVC',
        'SVR', 'NuSVC', 'NuSVR', 'BernoulliNB', 'GaussianNB', 'MultinomialNB', 'MLPClassifier',
        'MLPRegressor', 'AdaBoostClassifier', 'AdaBoostRegressor', 'VotingClassifier', 
        'VotingRegressor', 'StackingClassifier', 'StackingRegressor', 'Pipeline', 'FeatureUnion',
    ])

    # Model Evaluation-related classes and functions
    sklearn_keywords['Model_Evaluation']['sklearn'].extend([
        'accuracy_score', 'confusion_matrix', 'roc_auc_score', 'precision_score',
        'recall_score', 'f1_score', 'mean_squared_error', 'mean_absolute_error',
        'r2_score', 'explained_variance_score', 'log_loss', 'classification_report',
        'mean_squared_log_error', 'median_absolute_error', 'brier_score_loss',
        'precision_recall_curve', 'roc_curve', 'det_curve'
    ])

    # Model Interpretation-related classes and functions
    sklearn_keywords['Model_Interpretation']['sklearn'].extend([
        'permutation_importance', 'plot_partial_dependence', 'PartialDependenceDisplay',
        'decision_function', 'predict_proba', 'coef_', 'feature_importances_', 'plot_tree'
    ])

    # Hyperparameter Tuning-related classes and functions
    sklearn_keywords['Hyperparameter_Tuning']['sklearn'].extend([
        'GridSearchCV', 'RandomizedSearchCV', 'cross_val_score', 'KFold', 'StratifiedKFold',
        'train_test_split', 'validation_curve', 'learning_curve', 'GridSearchCV', 'RandomizedSearchCV',
        'ParameterGrid', 'ParameterSampler', 'HalvingGridSearchCV', 'HalvingRandomSearchCV'
    ])

    # Visualization-related classes and functions
    sklearn_keywords['Visualization']['sklearn'].extend([
        'plot_confusion_matrix', 'plot_roc_curve', 'plot_precision_recall_curve',
        'plot_partial_dependence', 'scatter_matrix', 'plot_tree', 'plot_tree_'
    ])

    # Debug-related classes and functions
    sklearn_keywords['Debug']['sklearn'].extend([
        'check_X_y', 'check_array', 'check_is_fitted', 'assert_all_finite', 'assert_raises',
        'assert_warns', 'assert_equal', 'assert_not_equal', 'assert_true', 'assert_false', 
    ])

    # Data Export-related classes and functions
    sklearn_keywords['Data_Export']['sklearn'].extend([
        'joblib.dump', 'joblib.load'
    ])

    # Other categories
    # sklearn_keywords['Other']['sklearn']

    return sklearn_keywords


def hardcoded_numpy_keywords():
    """
    Function to hardcode a dictionary with numpy classes and functions based on their functionality.

    :return (dict): dictionary with hardcoded keywords
    """

    numpy_keywords = {
        'Environment': {'numpy': []},
        'Data_Extraction': {'numpy': []},
        'Exploratory_Data_Analysis': {'numpy': []},
        'Data_Transform': {'numpy': []},
        'Model_Train': {'numpy': []},
        'Model_Evaluation': {'numpy': []},
        'Model_Interpretation': {'numpy': []},
        'Hyperparameter_Tuning': {'numpy': []},
        'Visualization': {'numpy': []},
        'Debug': {'numpy': []},
        'Data_Export': {'numpy': []},
        'Other': {'numpy': []},
    }

    # Environment-related classes and functions
    numpy_keywords['Environment']['numpy'].extend([
        'show_config', 'get_include', 'get_info', '.show', 'seterr', 'geterr', 'errstate',
        '_NoValueType', 'PytestTester', 'configuration', 'get_versions'
    ])

    # Data Extraction-related classes and functions
    numpy_keywords['Data_Extraction']['numpy'].extend([
        'loadtxt', 'genfromtxt', 'fromfile', 'frombuffer', 'fromstring', 'memmap', 'nditer',
        '_ndptr', 'fromiter', 'fromfunction', 'buildusevar', 'buildusevars'
    ])

    # Exploratory Data Analysis-related classes and functions
    numpy_keywords['Exploratory_Data_Analysis']['numpy'].extend([
        'np.mean', 'np.std', 'np.var', 'np.min', 'np.max', 'np.percentile', 'np.quantile', 'np.bincount',
        'np.histogram', 'np.unique', 'numpy.mean', 'numpy.std', 'numpy.var', 'numpy.min', 'numpy.max',
        'numpy.percentile', 'numpy.quantile', 'numpy.bincount', 'numpy.histogram', 'numpy.unique',
        'cumsum', 'cumprod', 'corrcoef', 'cov', '.average', '.median', '.nanmean', 
        '.nanstd', '.nanvar', '.nanmin', '.nanmax', '.nanpercentile', '.nanquantile', '.nanmedian'

    ])

    # Data Transformation-related classes and functions
    numpy_keywords['Data_Transform']['numpy'].extend([
        '.reshape', 'ravel', '.transpose', 'swapaxes', 'rollaxis', 'moveaxis', 'np.flip', 'np.rot90', 'np.resize',
        'pad', 'tile', 'repeat', '.split', 'array_split', 'hsplit', 'vsplit', 'dsplit', 'np.concatenate',
        'stack', 'hstack', 'vstack', 'dstack', 'column_stack', 'row_stack',
        'broadcast_to', 'einsum', 'np.dot', 'matmul', '.cross', 'np.outer', 'np.inner', 'tensordot'
    ])

    # Model Training-related classes and functions (specific to algorithms or machine learning, mostly empty for numpy)
    None

    # Model Evaluation-related classes and functions (mostly for metrics, error calculations)
    numpy_keywords['Model_Evaluation']['numpy'].extend([
        'mean_absolute_error', 'mean_squared_error', 'root_mean_squared_error', 'r2_score'
    ])

    # Model Interpretation-related classes and functions
    numpy_keywords['Model_Interpretation']['numpy'].extend([
        'gradient', 'hessian', 'jacobian'
    ])

    # Hyperparameter Tuning-related classes and functions
    None

    # Visualization-related classes and functions
    None

    # Debug-related classes and functions
    numpy_keywords['Debug']['numpy'].extend([
        'seterr', 'geterr', 'errstate', 'may_share_memory', 'shares_memory',
        'testing', 'assert_equal', 'assert_almost_equal', 'assert_array_equal',
        'assert_array_almost_equal', 'assert_raises', 'assert_warns'
    ])

    # Data Export-related classes and functions
    numpy_keywords['Data_Export']['numpy'].extend([
        'save', 'savez', 'savez_compressed', 'savetxt', 'tofile'
    ])

    # Other categories
    None

    return numpy_keywords


def hardcoded_pandas_keywords():
    """
    Function to hardcode a dictionary with pandas classes and functions based on their functionality.

    :return (dict): dictionary with hardcoded keywords
    """

    pandas_keywords = {
        'Environment': {'pandas': []},
        'Data_Extraction': {'pandas': []},
        'Exploratory_Data_Analysis': {'pandas': []},
        'Data_Transform': {'pandas': []},
        'Model_Train': {'pandas': []},
        'Model_Evaluation': {'pandas': []},
        'Model_Interpretation': {'pandas': []},
        'Hyperparameter_Tuning': {'pandas': []},
        'Visualization': {'pandas': []},
        'Debug': {'pandas': []},
        'Data_Export': {'pandas': []},
        'Other': {'pandas': []},
    }

    # Environment-related classes and functions
    pandas_keywords['Environment']['pandas'].extend([
        '.show_versions', '.set_option', '.get_option', '.reset_option', '.option_context',
        '.options', 'describe_optio', '_default_locale_getter', '_valid_locales', 'can_set_locale', 'get_locales', 'set_locale',
        'get_versions'
    ])

    # Data Extraction-related classes and functions
    pandas_keywords['Data_Extraction']['pandas'].extend([
        'read_csv', 'read_excel', 'read_json', 'read_html', 'read_sql', 'read_sql_table',
        'read_sql_query', 'read_pickle', 'read_clipboard', 'read_hdf', 'read_parquet',
        'read_orc', 'read_sas', 'read_spss', 'read_stata', 'read_feather', 'read_gbq',
        'read_xml', 'get_data_from_filepath', 'preprocess_data'
    ])

    # Exploratory Data Analysis-related classes and functions
    pandas_keywords['Exploratory_Data_Analysis']['pandas'].extend([ 
        'head', 'tail', 'info', 'describe', 'dtypes', 'columns', 'shape', 'size', 'memory_usage',
        '.describe', '.info', '.head', '.tail', '.sample', '.nunique', '.isna', '.isnull', '.notna',
        '.notnull', '.duplicated', '.corr', '.cov', '.skew', '.kurt', '.sum', '.mean', '.std',
        '.var', '.min', '.max', '.median', '.mode', '.value_counts', '.groupby'
    ])
    
    # Data Transformation-related classes and functions
    pandas_keywords['Data_Transform']['pandas'].extend([
        '.drop', '.dropna', '.fillna', '.replace', '.interpolate', '.ffill', '.bfill', '.sort_values',
        '.sort_index', '.reset_index', '.set_index', '.rename', '.apply', '.applymap', '.map', '.merge',
        '.join', '.concat', '.pivot', '.pivot_table', '.melt', '.stack', '.unstack', '.explode', '.shift',
        '.diff', '.pct_change', '.rolling', '.expanding', '.ewm', '.cut', '.qcut', '.get_dummies'
    ])  

    # Model Training-related classes and functions (mostly empty for pandas)
    None

    # Model Evaluation-related classes and functions (mostly empty for pandas)
    None

    # Model Interpretation-related classes and functions (mostly empty for pandas)
    None

    # Hyperparameter Tuning-related classes and functions (mostly empty for pandas)
    None

    # Visualization-related classes and functions
    pandas_keywords['Visualization']['pandas'].extend([
        'pd.plot', 'pd.plotting', 'pd.hist', 'pd.boxplot', 'pd.bar', 'pd.barh', 'pd.line', 'pd.scatter', 'pd.pie',
        'pandas.plot', 'pandas.plotting', 'pandas.hist', 'pandas.boxplot', 'pandas.bar', 'pandas.barh', 'pandas.line',
        'pandas.scatter', 'pandas.pie', 'pd.plotting.scatter_matrix', 'pd.plotting.parallel_coordinates',
        'pd.area', '.hexbin', '.kde', 'pd.density', '.plotting.scatter_matrix', '.plotting.parallel_coordinates',
        '.plotting.lag_plot', '.plotting.autocorrelation_plot', '.plotting.bootstrap_plot', '.plotting.table',
        '.plotting.scatter_matrix', '.plotting.andrews_curve', '.plotting.radviz'
    ])

    # Debug-related classes and functions
    pandas_keywords['Debug']['pandas'].extend([
        'pd.testing.', 'pandas.testing.', 'assert_frame_equal', 'assert_series_equal', 'assert_index_equal',
        'test_assert', 'assert_frame_equal', 'assert_series_equal', 'assert_index_equal',
        'decompress_file', 'ensure_clean', 'ensure_clean_dir', 'ensure_safe_environment_variables',
        'set_timezone'
    ])  

    # Data Export-related classes and functions
    pandas_keywords['Data_Export']['pandas'].extend([
        'to_csv', 'to_excel', 'to_json', 'to_html', 'to_latex', 'to_clipboard', 'to_sql', 'to_gbq',
        'to_hdf', 'to_feather', 'to_parquet', 'to_pickle', 'to_markdown', 'to_stata', 'to_spss',
        'to_sas', 'to_orc', 'to_string', 'to_string', 'to_latex', 'to_clipboard', 'to_sql', 'to_gbq',
        '.to_records', '.to_dict'
    ])

    # Other categories
    None

    return pandas_keywords


def hardcoded_tensorflow_keywords():
    """
    Function to hardcode a dictionary with tensorflow classes and functions based on their functionality.

    :return (dict): dictionary with hardcoded keywords
    """
        
    tensorflow_keywords = {
        'Environment': {'tensorflow': []},
        'Data_Extraction': {'tensorflow': []},
        'Exploratory_Data_Analysis': {'tensorflow': []},
        'Data_Transform': {'tensorflow': []},
        'Model_Train': {'tensorflow': []},
        'Model_Evaluation': {'tensorflow': []},
        'Model_Interpretation': {'tensorflow': []},
        'Hyperparameter_Tuning': {'tensorflow': []},
        'Visualization': {'tensorflow': []},
        'Debug': {'tensorflow': []},
        'Data_Export': {'tensorflow': []},
        'Other': {'tensorflow': []},
    }

    # Environment-related classes and functions
    tensorflow_keywords['Environment']['tensorflow'].extend([
        'get_logger', 'get_version', 'ConfigProto', 'Session', 'InteractiveSession',
        'reset_default_graph', 'placeholder', 'constant', 'variable', 'name_scope', 'device',
        'GPUOptions', 'RunConfig', 'ConfigProto', 'get_variable', 'get_collection', 'trainable_variables',
        'train.Saver', 'trainable_variables', 'checkpoint_management'
    ])

    # Data Extraction-related classes and functions
    tensorflow_keywords['Data_Extraction']['tensorflow'].extend([
        '.decode_json', '.decode_html', '.decode_sql', 'decode_pickle',
        'decode_clipboard', 'decode_hdf', 'decode_parquet', 'decode_orc', 'decode_sas', 'decode_spss',
        'decode_stata', 'decode_feather', 'decode_gbq', 'decode_xml', 'get_data_from_filepath',
        'preprocess_data', 'Dataset', '.make_one_shot_iterator', '.Iterator', '.TFRecordDataset', '.FixedLengthRecordDataset',
        '.TextLineDataset', '.experimental.make_csv_dataset', '.read_file', '.decode_csv', '.decode_raw',
        '.decode_image', '.decode_jpeg', '.decode_png', '.decode_gif'
    ])

    # Exploratory Data Analysis-related classes and functions	
    tensorflow_keywords['Exploratory_Data_Analysis']['tensorflow'].extend([
        '.reduce_mean', '.reduce_std', '.reduce_var', '.reduce_min', '.reduce_max', '.reduce_sum',
        '.reduce_prod', '.reduce_all', '.reduce_any', '.reduce_logsumexp', '.reduce_euclidean_norm',
        '.reduce_rms',  '.scalar', 'tf.histogram', 'tf.image', 'tf.audio', 'tf.text', 'tf.FileWriter',
        'tf.Summary', 'tf.merge_all', 'tf.merge', 'tf.initialize', '.tensor_summary',
        'tensorflow.histogram', 'tensorflow.image', 'tensorflow.audio', 'tensorflow.text', 'tensorflow.scalar',
        'tensorflow.tensor_summary', 'tensorflow.histogram_summary',
        'image_summary', 'audio_summary', 'text_summary', 'scalar_summary', 'histogram_summary',
    ])

    # Data Transformation-related classes and functions
    tensorflow_keywords['Data_Transform']['tensorflow'].extend([
        '.add', '.subtract', '.multiply', '.divide', '.log', '.exp', '.sqrt', '.pow',
        '.reduce_sum', '.reduce_mean', '.reduce_max', '.reduce_min', '.reduce_prod', '.reduce_all',
        '.reduce_any', '.square', '.maximum', '.minimum', '.sin', '.cos', '.tan', '.asin',
        '.acos', '.atan', '.atan2', '.sinh', '.cosh', '.tanh', '.asinh', '.acosh', '.atanh',
        '.equal', '.not_equal', '.less', '.less_equal', '.greater', '.greater_equal', '.logical_and',
        '.logical_or', '.logical_not', '.logical_xor', '.abs', '.negative', '.sign', '.reciprocal',
        '.round', '.ceil', '.floor', '.rint', '.is_nan', '.is_inf', '.mod', '.floormod', '.truncate',
        '.resize_s', '.resize', '.central_crop', '.crop_to_bounding_box', '.pad_to_bounding_box',
        '.extract_glimpse', '.extract_patches', '.flip_left_right', '.flip_up_down', '.transpose_',
        '.adjust_brightness', '.adjust_contrast', '.adjust_gamma', '.adjust_hue', '.adjust_saturation',
        '.per_image_standardization', '.convert_image_dtype'
    ])

    # Model Training-related classes and functions
    tensorflow_keywords['Model_Train']['tensorflow'].extend([
        'tf.estimator', 'tf.keras', 'tf.nn', 'tf.layers', 'tf.losses', 'tf.metrics', 'tf.optimizers',
        '.layers.Dense', '.layers.Conv2D', '.layers.MaxPooling2D', '.layers.AveragePooling2D', '.layers.GlobalMaxPooling2D',
        '.layers.GlobalAveragePooling2D', '.layers.Flatten', '.layers.Dropout', '.layers.BatchNormalization',
        '.layers.Activation', '.layers.LeakyReLU', '.layers.ReLU', '.layers.Softmax', '.layers.LSTM',
        '.layers.GRU', '.layers.SimpleRNN', '.layers.Bidirectional', '.layers.TimeDistributed', '.layers.Embedding',
        '.layers.Input', '.layers.InputLayer', '.layers.Add', '.layers.Subtract', '.layers.Multiply', '.layers.Average',
        '.layers.Maximum', '.layers.Minimum', '.layers.Concatenate', '.layers.Dot', '.layers.Permute', '.layers.Reshape',
        '.layers.Lambda', '.layers.RepeatVector', '.layers.Cropping1D', '.layers.Cropping2D', '.layers.Cropping3D',
        '.layers.UpSampling1D', '.layers.UpSampling2D', '.layers.UpSampling3D', '.layers.ZeroPadding1D', '.layers.ZeroPadding2D',
        '.layers.ZeroPadding3D', '.optimizers.Adam', '.optimizers.SGD', '.optimizers.RMSprop', '.optimizers.Adagrad',
        '.optimizers.Adadelta', '.optimizers.Adamax', '.optimizers.Nadam', '.optimizers.Ftrl', 'train.GradientDescentOptimizer',
        'train.AdamOptimizer', 'train.AdagradOptimizer', 'train.RMSPropOptimizer', 'train.MomentumOptimizer', 'train.AdagradDAOptimizer', 'train.ProximalGradientDescentOptimizer',
        'train.ProximalAdagradOptimizer', 'train.FtrlOptimizer', 'train.AdadeltaOptimizer', 'train.AdamaxOptimizer', 'train.NadamOptimizer'
    ])

    # Model Evaluation-related classes and functions
    tensorflow_keywords['Model_Evaluation']['tensorflow'].extend([
        '.mean_absolute_error', '.mean_squared_error', '.mean_squared_logarithmic_error', '.median_absolute_error',
        '.mean_absolute_percentage_error', '.mean_squared_percentage_error', '.root_mean_squared_error', '.r2_score',
        '.accuracy', '.binary_accuracy', '.categorical_accuracy', '.sparse_categorical_accuracy',
        '.top_k_categorical_accuracy', '.sparse_top_k_categorical_accuracy', '.hinge', '.squared_hinge',
        '.categorical_crossentropy', '.sparse_categorical_crossentropy', '.binary_crossentropy',
        '.kullback_leibler_divergence', '.poisson', '.cosine_proximity', '.auc', '.precision',
        '.recall', '.f1_score', '.f1_score_macro', '.f1_score_micro', '.f1_score_weighted',
        '.mean_squared_error', '.root_mean_squared_error', '.mean_absolute_error', '.mean_absolute_percentage_error'
    ])

    # Model Interpretation-related classes and functions
    tensorflow_keywords['Model_Interpretation']['tensorflow'].extend([
        '.gradient', '.hessian', '.jacobian', '.gradients', '.hessians', '.jacobians', '.gradients', '.hessians', '.jacobians'
        '.utils.vis_utils.plot_model', '.utils.vis_utils.model_to_dot', 'summary', '.utils.vis_utils.visualize_model',
        '.utils.vis_utils.visualize_layer', '.utils.vis_utils.visualize_weights', '.utils.vis_utils.visualize_activations',
        '.utils.vis_utils.visualize_filters'
    ])

    # Hyperparameter Tuning-related classes and functions
    tensorflow_keywords['Hyperparameter_Tuning']['tensorflow'].extend([
        '.GridSearchCV', '.RandomizedSearchCV', '.cross_val_score', '.KFold', '.StratifiedKFold',
        '.train_test_split', '.validation_curve', '.learning_curve', '.GridSearchCV', '.RandomizedSearchCV',
        '.ParameterGrid', '.ParameterSampler', '.HalvingGridSearchCV', '.HalvingRandomSearchCV'
    ])

    # Visualization-related classes and functions
    tensorflow_keywords['Visualization']['tensorflow'].extend([
        '.plot_confusion_matrix', '.plot_roc_curve', '.plot_precision_recall_curve',
        '.plot_partial_dependence', '.scatter_matrix', '.plot_tree', '.plot_tree_'
        '.plot_model', '.model_to_dot', '.visualize_model', '.visualize_layer',
        '.visualize_weights', '.visualize_activations', '.visualize_filters'
    ])

    # Debug-related classes and functions
    tensorflow_keywords['Debug']['tensorflow'].extend([
        'debugging',
        '.check_X_y', '.check_array', '.check_is_fitted', '.assert_all_finite', '.assert_raises',
        '.assert_warns', '.assert_equal', '.assert_not_equal', '.assert_true', '.assert_false', 
    ])

    # Data Export-related classes and functions
    tensorflow_keywords['Data_Export']['tensorflow'].extend([
        'train.Saver', 'saved_model', '.simple_save', '.write_file',
        '.write_graph', '.write_records'
    ])
    

    return tensorflow_keywords


def hardcoded_torch_keywords():
    """
    Function to hardcode a dictionary with pytorch classes and functions based on their functionality.

    :return (dict): dictionary with hardcoded keywords
    """
    
    pytorch_keywords = {
        'Environment': {'torch': []},
        'Data_Extraction': {'torch': []},
        'Exploratory_Data_Analysis': {'torch': []},
        'Data_Transform': {'torch': []},
        'Model_Train': {'torch': []},
        'Model_Evaluation': {'torch': []},
        'Model_Interpretation': {'torch': []},
        'Hyperparameter_Tuning': {'torch': []},
        'Visualization': {'torch': []},
        'Debug': {'torch': []},
        'Data_Export': {'torch': []},
        'Other': {'torch': []},
    }

    # Environment-related
    pytorch_keywords['Environment']['torch'].extend([
        '.__config__._cxx_flags', '.__config__.parallel_info', '.__config__.show',
        '._appdirs.AppDirs', '._awaits._PyAwaitMeta', '._compile._disable_dynamo',
        '._custom_op.impl.CustomOp', '._custom_op.impl.FuncAndLocation',
        '._custom_op.impl._custom_op_with_schema', '._custom_op.impl._find_custom_op',
        '._custom_op.impl.custom_op', '._custom_op.impl.custom_op_from_existing',
        '._custom_op.impl.derived_types', '._custom_ops._destroy', '._custom_ops.custom_op',
        '._custom_ops.impl', '._custom_ops.impl_abstract', '._custom_ops.impl_backward'
    ])

    # Data Extraction
    pytorch_keywords['Data_Extraction']['torch'].extend([
        '.DataLoader', '.Dataset', '.TensorDataset', '.ConcatDataset', '.random_split', '.ImageFolder'
    ])

    # Exploratory Data Analysis
    pytorch_keywords['Exploratory_Data_Analysis']['torch'].extend([
        'torch.mean', 'torch.std', 'torch.sum', 'torch.min', 'torch.max'
    ])

    # Data Transformation
    pytorch_keywords['Data_Transform']['torch'].extend([
        '.Compose', '.ToTensor', '.Normalize', '.Resize', '.RandomCrop'
    ])

    # Model Training
    pytorch_keywords['Model_Train']['torch'].extend([
        '.Module', '.Sequential', '.Linear', '.Conv2d',
        '.ReLU', '.optim.Adam', '.optim.SGD', '.optim.lr_scheduler.StepLR',
        '.optim.lr_scheduler.MultiStepLR', '.optim.lr_scheduler.ExponentialLR',
    ])

    # Model Evaluation
    pytorch_keywords['Model_Evaluation']['torch'].extend([
        '.CrossEntropyLoss', '.MSELoss', '.BCELoss',
        '.functional.softmax', '.functional.sigmoid', 'metrics.Accuracy'
    ])

    # Model Interpretation
    pytorch_keywords['Model_Interpretation']['torch'].extend([
        '.resnet50', '.vgg16', '.mobilenet_v2', '.densenet121', '.inception_v3'
    ])

    # Hyperparameter Tuning
    pytorch_keywords['Hyperparameter_Tuning']['torch'].extend([
        '.StepLR', '.MultiStepLR', '.ExponentialLR', '.ReduceLROnPlateau'
    ])

    # Visualization
    pytorch_keywords['Visualization']['torch'].extend([
        '.make_grid', '.ToPILImage'
    ])

    # Debug
    pytorch_keywords['Debug']['torch'].extend([
        '.set_detect_anomaly', '.autograd.profiler.profile',
        '.autograd.gradcheck', '.bottleneck'
    ])

    # Data Export
    pytorch_keywords['Data_Export']['torch'].extend([
        'torch.save', 'torch.load', 'torch.jit.trace', 'torch.jit.script', 'torch.onnx.export'
    ])

    return pytorch_keywords


def get_keywords(keywords, method="all"):
    """
    Function to get the keywords for the classification of a notebook.

    :param method (str): method to use to get the keywords. Options are

    :return (dict): dictionary with the keywords for the classification of a notebook
    """

    class_keywords = {key: {} for key in keywords}

    if method.upper() == "HARDCODED":
        return hardcoded_keywords()
    elif method.upper() == "SKLEARN":
        sklearn_keywords = hardcoded_sklearn_keywords()
        return merge_keyword_dict(hardcoded_keywords, sklearn_keywords)
    elif method.upper() == "NUMPY":
        numpy_keywords = hardcoded_numpy_keywords()
        return merge_keyword_dict(hardcoded_keywords, numpy_keywords)
    elif method.upper() == "PANDAS":
        pandas_keywords = hardcoded_pandas_keywords()
        return merge_keyword_dict(hardcoded_keywords, pandas_keywords)
    elif method.upper() == "TENSORFLOW":
        tensorflow_keywords = hardcoded_tensorflow_keywords()
        return merge_keyword_dict(hardcoded_keywords, tensorflow_keywords)
    elif method.upper() == "TORCH":
        torch_keywords = hardcoded_torch_keywords()
        return merge_keyword_dict(hardcoded_keywords, torch_keywords)
    elif method.upper() == "ALL":
        temp_keywords = hardcoded_keywords()
        temp_keywords = merge_keyword_dict(temp_keywords, hardcoded_sklearn_keywords())
        temp_keywords = merge_keyword_dict(temp_keywords, hardcoded_numpy_keywords())
        temp_keywords = merge_keyword_dict(temp_keywords, hardcoded_pandas_keywords())
        temp_keywords = merge_keyword_dict(temp_keywords, hardcoded_tensorflow_keywords())
        temp_keywords = merge_keyword_dict(temp_keywords, hardcoded_torch_keywords())
        return temp_keywords
    elif method.upper() == "ALL_NO_HARDCODED":
        temp_keywords = hardcoded_sklearn_keywords()
        temp_keywords = merge_keyword_dict(temp_keywords, hardcoded_numpy_keywords())
        temp_keywords = merge_keyword_dict(temp_keywords, hardcoded_pandas_keywords())
        temp_keywords = merge_keyword_dict(temp_keywords, hardcoded_tensorflow_keywords())
        temp_keywords = merge_keyword_dict(temp_keywords, hardcoded_torch_keywords())
        return temp_keywords	

    else:
        raise ValueError("Method not recognized. Please use 'hardcoded', 'sklearn', 'numpy', 'pandas', 'tensorflow', 'torch', 'all', or 'all_no hardocded'.")   
        return class_keywords


def merge_keyword_dict(dict1, dict2):
    """
    Function to merge two dictionaries of dictionaries based on a common key in second dictionary.

    :param dict1 (dict): dictionary to merge into
    :param dict2 (dict): dictionary to merge from

    :return (dict): merged dictionary
    """
    
    new_dict = dict1.copy()
    for key, value in dict1.items(): # per class
        for sub_key, sub_value in value.items(): # per method
            if sub_key in dict2[key]:
                new_dict[key][sub_key].extend(dict2[key][sub_key])
    
    for key, value in dict2.items():
        for sub_key, sub_value in value.items():
            if sub_key not in dict1[key]:
                new_dict[key][sub_key] = sub_value

    return new_dict


def classify_by_keywords(source, class_keywords):
    """
    Function to classify a cell based on the keywords.

    :param source (str): source code of the cell
    :param keywords (dict): dictionary with the keywords for the classification of a notebook

    :return class_probability (dict): dictionary with the probabilities for the classification of a cell
    """

    # initialize variables
    class_probability = {key: 0 for key in class_keywords}
    detailed_scores = {classification: {} for classification in class_keywords}
    class_scores = {}
    class_scores_normalized_keyword_count = {}
    

    # check empty cell
    if source is None:
        return "Other", class_probability, []
    if isinstance(source, list):
        source = "\n".join(source)
    source = source.lower() # lowercase all the source code


    # Function to collect keywords in usefull structure
    def collect_all_keywords(classification, details): #TODO: also do this earlier?
        keywords = []
        for key in details:
            # lowercase all the keywords, as source is also lowercased
            sub_keys = [sub_key.lower() for sub_key in details[key]]
            keywords.extend(sub_keys)
        return keywords


    # count keyword occurence in source
    for classification, details in class_keywords.items():
        keywords = collect_all_keywords(classification, details)
        score = sum(source.count(keyword) for keyword in keywords)
        class_scores[classification] = score
        class_scores_normalized_keyword_count[classification] = score / len(keywords)
        

        # add keywords to detailed_scores
        for keyword in keywords:
            count = source.count(keyword)
            if count > 0:
                if keyword not in detailed_scores[classification]:
                    detailed_scores[classification][keyword] = count #TODO: change to 1 ?
                else:
                    detailed_scores[classification][keyword] += count #TODO: change to 1 ?
            
            # extra rule environment
            if keyword == "import " and count > 3:
                break


    # Determine the classification with the highest keyword count
    max_classification = max(class_scores, key=class_scores.get)
    max_score = class_scores[max_classification]


    # return probability dictionary per class, and the classification with the highest probability, and the keywords found
    if max_score > 0:
        # calculate the probability of each classification
        class_probability = {classification: score / max_score for classification, score in class_scores.items()}
        return max_classification, class_probability, detailed_scores
    else:
        return "Other", class_probability, []
    
