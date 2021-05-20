---
title: "DengAI: Predicting Disease Spread - Imputation and Stationarity Problems"
categories:
  - Time Series
  - Python
header:
    image: /assets/article_images/dengai/cover1.png
toc: true
---

<em> [Github Repository](https://medium.com/r/?url=https%3A%2F%2Fgithub.com%2Fdata4help%2Fdengai) </em>

One of the biggest data challenge on DrivenData, with more than 9000 participants is the [DengAI challenge](https://medium.com/r/?url=https%3A%2F%2Fwww.drivendata.org%2Fcompetitions%2F44%2Fdengai-predicting-disease-spread%2F). The objective of this challenge is predict the number of dengue fever cases in two different cities.

This blogpost series covers our journey of tackling this problem, starting from initial data analysis, imputation and stationarity problems up un to the different forecasting attempts. This first post covers the imputation and stationarity checks for both cities in the challenge, before moving on to trying different forecasting methdologies.

Throughout this post, code-snippets are shown in order to give an understanding of how the concepts discussed are implemented into code. The entire Github repository for the imputation and stationary adjustment can be found here.

Furthermore, in order to ensure readability we decided to show graphs only for the city San Jose instead showing it for both cities.

## Imputation

Imputation describes the process of filling missing values within a dataset. Given the wide range of possibilities for imputation and the severe amount of missing data within this project, it is worthwhile to go over some of the methods and empirically check which one to use.

Overall, we divide all imputation methods into the two categories: basic and advanced. With basic methods we mean off the shelf, quick imputation methods, which are oftentimes already build into Pandas. Advanced imputation methods deal with model-based approaches where the missing values are attempted to be predicted, using the remaining columns.

Given that the model-based imputation methods normally result in superior performance, the question might arise why we do not simply use the advanced method for all columns. The reason for that is that our dataset has several observations where all features are missing. The presence of these observations make multivariate imputation methods impossible.

We therefore divide the features into two categories. All features, or columns, which have fewer than 1 percent missing observations are imputed using more basic methods, whereas model-based approaches are used for features which exhibit more missing observations than this threshold.

The code snippet below counts the percentage of missing observations, divides all features into one of two aforementioned categories and creates a graph to visualize the results.

```python
def pct_missing_data(self, data, cols, threshold, city_name):
        """
        This method does two things. First, it creates a chart showing the
        percentages of missing data. Second, it returns which columns
        have less than a certain threshold percentage of data, and which
        columns have more.

        Parameters
        ----------
        data : DataFrame
            DataFrame containing all the data
        cols : list
            List containing all the columns we have to fill
        threshold : int
            The threshold which divides the easy and difficult columns
        city_name : str
            A string containing the city name for which we calculate

        Returns
        -------
        easy_columns : list
            List containing the columns which have less then the threshold
            missing data
        diff_columns : list
            List containing the columns which have more than the threshold
            missing data

        """

        # Calculating the percentage missing
        num_of_obs = len(data)
        num_of_nans = data.loc[:, cols].isna().sum()
        df_pct_missing = pd.DataFrame(num_of_nans
                                      / num_of_obs*100).reset_index()
        df_pct_missing.rename(columns={"index": "columns",
                                       0: "pct_missing"}, inplace=True)
        df_sorted_values = df_pct_missing.sort_values(by="pct_missing",
                                                      ascending=False)

        # Column division
        bool_easy = df_sorted_values.loc[:, "pct_missing"] < threshold
        easy_columns = df_sorted_values.loc[bool_easy, "columns"]
        diff_columns = list(set(cols) - set(easy_columns))

        # Plotting the data
        fig, axs = plt.subplots(figsize=(20, 10))
        axs.bar(df_sorted_values.loc[:, "columns"],
                df_sorted_values.loc[:, "pct_missing"])
        axs.tick_params(axis="both", labelsize=16, labelrotation=90)
        axs.set_ylabel("Percentage of missing observations", fontsize=18)
        axs.set_xlabel("Variables", fontsize=18)
        axs.axhline(threshold, linestyle="dashed", color="red",
                    label="{} percent threshold".format(threshold))
        axs.legend(prop={"size": 20})
        fig.savefig(r"{}/{}_miss_data_pct.png".format(output_path, city_name),
                    bbox_inches="tight")

        return easy_columns, diff_columns
```

The resulting graph below shows four features which have more than 1 percent of observations missing. Especially the feature ndvi_ne, which describes satellite vegetation in the north-west of the city has a severe amount of missing data, with more around 20% of all observation missing.

![](/assets/post_images/dengai/picture1_1.png)

All imputation methods applied are compared using the [normalized root mean squared error](http://normalized%20root%20mean%20squared%20error) (NRMSE). We use this quality estimation method because of its capability for making variables with different scales comparable. Given that the NRMSE is not directly implemented in Python, we use the following snippet to implement it.

```python
def nrmse(self, y_true, y_pred, n):
    """
    This function calculates the normalized root mean squared error.

    Parameters
    ----------
    y_true : array
        The true values
    y_pred : array
        The predictions
    n : int
        The number of rows we testing for performance

    Returns
    -------
    rounded_nrmse : float
        The resulting, rounded nrmse

    """
    ts_min, ts_max = np.min(y_true), np.max(y_true)
    mse = sum((y_true-y_pred)**2) / n
    nrmse_value = np.sqrt(mse) / (ts_max-ts_min)
    rounded_nrmse = np.round(nrmse_value, 2)
    return rounded_nrmse
```

## Basic imputation methods

Python, and in particular the library Pandas, has multiple off-the-shelf imputation methods available. Arguably the most basic ones are forward fill (ffill) and backward fill (bfill), where we simply set the missing valueequal to the prior value (ffill) or to the proceeding value (bfill).

Other methods include the linear or cubic (the Scipy package also includes higher power if wanted) interpolation around a missing observation.

Lastly, we can use the average of the k nearest neighbours of a missing observations. For this problem we took the preceding and proceeding four observations of a missing observation and imputed it with the average of these eight values. This is not a build in method and therefore defined by us in the following way:

```python
def knn_mean(self, ts, n):
      """
      This function calculates the mean value of the n/2 values before
      and after it. This approach is therefore called the k nearest
      neighbour approach.

      Parameters
      ----------
      ts : array
          The time series we would like to impute
      n : int
          The number of time period before + after we would like
          to take the mean of

      Returns
      -------
      out : array
          The filled up time series.

      """
      out = np.copy(ts)
      for i, val in enumerate(ts):
          if np.isnan(val):
              n_by_2 = np.ceil(n/2)
              lower = np.max([0, int(i-n_by_2)])
              upper = np.min([len(ts)+1, int(i+n_by_2)])
              ts_near = np.concatenate([ts[lower:i], ts[i:upper]])
              out[i] = np.nanmean(ts_near)
      return out
```

Now it is time to apply and compare of these methods. We do that by randomly dropping 50 observations of all columns, which are afterwards imputed by all before mentioned methods. Afterwards we assess each method's performance through their NRMSE score. All of that, and the graphing of the results is done through the following code snippet.

```python
def imputation_table(self, data, cols, city_name):
        """
        This method calculates the nrmse for all columns and inserts them
        in a table. Additionally a graph is plotted in order for visual
        inspection afterwards. The score is calculated by randomly dropping
        50 values and then imputing them. Afterwards the performance is
        assessed.

        Parameters
        ----------
        data : DataFrame
            Dataframe which includes all columns
        cols : list
            List of columns we would like to impute
        city_name : str
            In order to know which city data was used, we specify the name

        Returns
        -------
        nrmse_df : DataFrame
            The results of each method for each column.

        """

        nrmse_df = pd.DataFrame(index=cols)
        print("Create imputation table")
        for col in tqdm(cols):

            original_series = data.loc[:, col]
            time_series = original_series.dropna().reset_index(drop=True)

            n = 50
            random.seed(42)
            rand_num = random.sample(range(0, len(time_series)), n)

            time_series_w_nan = time_series.copy()
            time_series_w_nan[rand_num] = np.nan

            # Forward fill ----
            ts_ffill = time_series_w_nan.ffill()
            nrmse_df.loc[col, "ffill"] = self.nrmse(time_series, ts_ffill, n)

            # Backward fill ----
            ts_bfill = time_series_w_nan.bfill()
            nrmse_df.loc[col, "bfill"] = self.nrmse(time_series, ts_bfill, n)

            # Linear Interpolation ----
            ts_linear = time_series_w_nan.interpolate(method="linear")
            nrmse_df.loc[col, "linear"] = self.nrmse(time_series,
                                                     ts_linear, n)

            # Cubic Interpolation ----
            ts_cubic = time_series_w_nan.interpolate(method="cubic")
            nrmse_df.loc[col, "cubic"] = self.nrmse(time_series, ts_cubic, n)

            # Mean of k nearest neighbours ----
            ts_knn = self.knn_mean(time_series_w_nan, 8)
            nrmse_df.loc[col, "knn"] = self.nrmse(time_series, ts_knn, n)

        # Plotting results
        adj_df = nrmse_df.reset_index()
        long_format = pd.melt(adj_df, id_vars=["index"], var_name=["nrmse"])
        fig, axs = plt.subplots(figsize=(20, 10))
        sns.barplot(x="index", y="value", hue="nrmse",
                    data=long_format, ax=axs)
        axs.tick_params(axis="both", labelsize=16, labelrotation=90)
        axs.set_ylabel("Normalized Root Mean Squared Root Error", fontsize=18)
        axs.set_xlabel("Variables", fontsize=18)
        axs.legend(prop={"size": 20})
        fig.savefig(r"{}/{}_imput_performance.png".format(output_path,
                                                          city_name),
                    bbox_inches="tight")

        return nrmse_df
```
The resulting graph below clearly shows which method is to be favored, namely the k nearest neighbours approach. The linear method also performs well, even though not as well as the knn method. The more naive methods like ffill and bfill do not perform as strongly.

![](/assets/post_images/dengai/picture1_2.png)

Afterwards, we impute all features which had fewer observations missing than our threshold of one percent. That means all features except the first four. The code below selects the best method for each column and afterwards imputes all actual missing values.

```python
def fill_by_method(self, original_series, method):
    """
    After we know what the best method is for each column, we would
    like to impute the missing values. This function lists all
    potential methods, except the model build one.
    Parameters
    ----------
    original_series : array
        The original array with all its missing values
    method : str
        A string describing the best working method
    Returns
    -------
    time_series : array
        The original array now filled the missing values with the
        method of choice
    """

    if method == "ffill":
        time_series = original_series.ffill()
    elif method == "bfill":
        time_series = original_series.bfill()
    elif method == "linear":
        time_series = original_series.interpolate(method="linear")
    elif method == "cubic":
        time_series = original_series.interpolate(method="cubic")
    elif method == "knn":
        time_series = self.knn_mean(original_series, 8)
    return time_series

def fill_easy_columns(self, data, easy_columns, nrmse_df):
    """
    This method goes through all easy declared columns and fills them
    up
    Parameters
    ----------
    data : Dataframe
        DataFrame containing all columns
    easy_columns : list
        List of all columns which can undergo the easy imputation
    nrmse_df : DataFrame
        Dataframe which contains the performance metrices of
        all imputation methods
    Returns
    -------
    data : Dataframe
        Dataframe with imputated columns
    """

    print("Filling easy columns")
    for col in tqdm(easy_columns):
        time_series = data.loc[:, col]
        best_method = nrmse_df.loc[col, :].sort_values().index[0]
        ts_filled = self.fill_by_method(time_series, best_method)
        data.loc[:, col] = ts_filled

        assert sum(data.loc[:, col].isna()) == 0, \
            "Easy imputation went wrong"
    return data
```

## The potential flaws of the knn approach

Unfortunately, the superior performance of the knn model comes with a price. For some features, we do not have a only one observation missing at a time, but multiple consecutively missing observations.

If for example we have 12 consecutive missing observations, the knn method cannot calculate any average out of the preceding and proceeding four observations, given that they are missing as well.

The image below, which was created with the beatiful missingno package, shows us that all four columns which were classified as being above our one percent threshold have at one point 15 consecutive missing observations. This makes it impossible to use the knn method for these columns and is the reason why we cannot use this imputation method for the heavily sparse columns.

![](/assets/post_images/dengai/picture1_3.png)

## Model-based imputation methods

The model-based imputation methods use, as already described earlier, the column with the missing observations as the target and uses all other possible columns as the features. After imputing all columns with fewer than one percent missing observations, we can now use all of them as features.

The model we are using is a RandomForestRegressor because of its good handling of noisy data. The code snippet below which hyperparameters were gridsearched.

```python
imputation_model = {
        "model": RandomForestRegressor(random_state=28),
        "param": {
            "n_estimators": [100],
            "max_depth": [int(x) for x in
                          np.linspace(10, 110, num=10)],
            "min_samples_split": [2, 5, 10, 15, 100],
            "min_samples_leaf": [1, 2, 5, 10]
            }
        }
```
We now run all four columns through the model-based approach and compare their performance to all aforementioned basic imputation methods. The following code snippet takes care of exactly that.

```python
def fill_diff_columns(self, data, model, diff_columns,
                          easy_columns, nrmse_df, city_name):
        """
        This method imputes the difficult columns. Difficult means that
        these columns miss more than the specified threshold percentage
        of observations. Because of that a model based approach is tried.
        If this approach proves better than the normal methods, it is
        applied.
        Furthermore, we plot the nrmse of the model based approach in order
        to compare these with the normal methods
        Parameters
        ----------
        data : DataFrame
            Dataframe containing all data
        model : dictionary
            Here we specify the model and the parameters we would like to try
        diff_columns : list
            List of columns we would like to try
        easy_columns : list
            List of columns which have less than the threshold percentage
            data missing
        nrmse_df : Dataframe
            Dataframe with the nrmse for all methods and columns
        city_name : str
            String specifying which city we are talking about
        Returns
        -------
        data : Dataframe
            Dataframe with imputated columns
        diff_nrmse_df : Dataframe
            Dataframe showing the nrmse performance of the difficult
            columns and all methods
        """
        non_knn_method = list(set(nrmse_df.columns) - set(["knn"]))
        diff_nrmse_df = nrmse_df.loc[diff_columns, non_knn_method]
        print("Filling difficult columns")
        for col in tqdm(diff_columns):

            # Getting data ready
            time_series = data.loc[:, col]
            non_nan_data = data.dropna(subset=[col])
            features = non_nan_data.loc[:, easy_columns]
            scaler = StandardScaler().fit(features)
            scaled_features = scaler.transform(features)
            target = non_nan_data.loc[:, col]

            # Model building and evaluation
            model_file_name = "{}/{}_{}_model.pickle".format(data_path,
                                                             city_name,
                                                             col)
            if not os.path.isfile(model_file_name):
                model_info = model_train(model, scaled_features,
                                         target,
                                         "neg_mean_squared_error",
                                         False)
                with open(model_file_name, "wb") as file:
                    pickle.dump(model_info, file)
            else:
                with open(model_file_name, "rb") as f:
                    model_info = pickle.load(f)
            target_min, target_max = np.min(target), np.max(target)
            rmse = np.sqrt(abs(model_info["scores"][0]))
            nrmse_value = rmse / (target_max-target_min)
            diff_nrmse_df.loc[col, "model"] = nrmse_value

            # Imputing the difficult ones
            argmin_method = np.argmin(diff_nrmse_df.loc[col, :])
            best_method = diff_nrmse_df.columns[argmin_method]
            bool_target_nan = time_series.isna()
            if best_method == "model":
                features = data.loc[bool_target_nan, easy_columns]
                scaled_features = scaler.transform(features)
                pred = model_info["model"].predict(scaled_features)
                data.loc[bool_target_nan, col] = pred
            else:
                pred = self.fill_by_method(time_series, best_method)
                data.loc[bool_target_nan, col] = pred

        assert data.loc[:, list(easy_columns) + diff_columns]\
            .isna().any().any() == False, "Still missing data"

        # Plotting results
        adj_df = diff_nrmse_df.reset_index()
        long_format = pd.melt(adj_df, id_vars=["index"], var_name=["nrmse"])
        fig, axs = plt.subplots(figsize=(20, 10))
        sns.barplot(x="index", y="value", hue="nrmse",
                    data=long_format, ax=axs)
        axs.tick_params(axis="both", labelsize=16, labelrotation=90)
        axs.set_ylabel("Normalized Root Mean Squared Root Error", fontsize=18)
        axs.set_xlabel("Variables", fontsize=18)
        axs.legend(prop={"size": 20})
        fig.savefig(r"{}/{}_diff_columns.png".format(output_path, city_name),
                    bbox_inches="tight")

        return data, diff_nrmse_df
```

Below we can see that our work was worthwhile. For three out of four columns we find a superior performance of the model-based approach compared to the basic imputation methods. We are now left with a fully imputed dataset with which we can proceed.

![](/assets/post_images/dengai/picture1_4.png)


## Stationarity Problems - Seasonality and Trend

In contrast to cross-sectional data, time series data comes with a whole bunch of different problems. Undoubtedly one of the biggest issues is the problem of stationarity. Stationarity describes a measure of regularity. It is this regularity which we depend on to exploit when building meaningful and powerful forecasting models. The absence of regularity makes it difficult at best to construct a model.

There are two types of stationarity, namely strict and covariance stationarity. In order for a time series to be fulfil strict stationarity, the series needs to be time independent. That would imply that the relationship between two observations of a series is only driven by the timely gap between them, but not on the time itself. This assumption is difficult, if not impossible for most time series to meet and therefore more focus is drawn on covariance stationarity.

For a time series to be covariance stationary, it is required that the unconditional first two moments, so the mean and variance, are finite and do not change with time. It is important to note that the time series is very much allowed to have a varying conditional mean. Additionally, it is required that the auto-covariance of a time series is only depending on the lag number, but not on the time itself. All these requirements are also stated below.

![](/assets/post_images/dengai/picture1_5.png)

There are many potential reasons for a time series to be non-stationary, including seasonalities, unit roots, deterministic trends and structural breaks. In the following section we will check and adjust our exogenous variable for each of these criteria to ensure stationarity and superior forecasting behavior.

## Seasonality

Seasonality is technically a form of non-stationarity because the mean of the time series is dependent on time factor. An example would be the spiking sales of a gift-shop around Christmas. Here the mean of the time series is explicitly dependent on time.

In order to adjust for seasonality within our exogenous variables, we first have to find out which variables actually exhibits that kind of behavior. This is done by applying a Fourier Transform. A Fourier transform disentangles a signal into its different frequencies and assesses the power of each individual frequency. The resulting plot, which shows power as a function of frequency is called a power spectrum. The frequency with the strongest power could then be potentially the driving seasonality in our time series. More information about Fourier transform and signal processing in general can be read up on an earlier blogpost of ours here.

The following code allows us to take a look into the power-plots of our 20 exogenous variable.

```python
def spike_finder(self, data, cols, city_name):
        """
        This method calculates the power-plots for all specified
        variables. Afterwards spikes above a certain threshold and
        which exhibit the desired prominence are marked. Afterwards
        an image of all columns is saved
        Parameters
        ----------
        data : DataFrame
            Dataframe containing all the columns for which we would
            like to calculate the power-plots of
        cols : list
            Columns which we would like to examine
        city_name : str
            A string denoting which city we are looking at
        Returns
        -------
        spikes_dict : dict
            Dictionary which saves the dominant and prominent
            frequencies for each exogenous variables
        """
        fig, axs = plt.subplots(nrows=4, ncols=5, figsize=(20, 20),
                                sharex=True)
        plt.subplots_adjust(right=None)
        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor="none", top=False, bottom=False,
                        left=False, right=False)
        plt.grid(False)
        plt.xlabel("Frequency [1 / Hour]", fontsize=22, labelpad=20)
        plt.ylabel("Amplitude", fontsize=22, labelpad=50)
        spikes_dict = {}
        axs = axs.ravel()
        for i, col in enumerate(cols):

            signal = data.loc[:, col].copy()
            fft_output = fft.fft(signal.values)
            power = np.abs(fft_output)
            freq = fft.fftfreq(len(signal))

            mask = freq > 0
            pos_freq = freq[mask]
            power = power[mask]

            axs[i].plot(pos_freq, power)
            axs[i].tick_params(axis="both", labelsize=16)
            axs[i].set_title(col, fontsize=12)
            
            relevant_power = power[:int(len(power)/4)]
            prominence = np.mean(relevant_power) * 5
            threshold = np.mean(relevant_power) + 3 * np.std(relevant_power)
            peaks = sig.find_peaks(relevant_power, prominence=prominence,
                                   threshold=threshold)[0]
            peak_freq = pos_freq[peaks]
            peak_power = power[peaks]
            axs[i].plot(peak_freq, peak_power, "ro")
            if len(peak_freq) > 0:
                spikes_dict[col] = (1/peak_freq).tolist()[0]
            else:
                spikes_dict[col] = np.nan
```

The plot below shows the resulting 20 exogenous variables. Whether or not a predominant and significant threshold is met for a variable is indicated by a red dot on top of a spike. If a red dot is visible, that means that the time series has a significantly driving frequency and therefore a strong seasonality component.

![](/assets/post_images/dengai/picture1_6.png)

One possibility to cross-check the results of the Fourier Transforms is to plot the Autocorrelation function. If we would try have a seasonality of order X, we would expect a significant correlation with lag X. The following snippet of code plots the autocorrelation function for all features and highlights those features which are found to have a seasonal affect according to the Fourier Transform.

```python
def acf_plots(self, data, cols, spike_dict, city_name):
    """
    This method plots the autocorrelation functions for all
    specified columns in a specified dataframe. Furthermore,
    the biggest possible spike for each column, if there is any,
    is made visible through a vertical line and a legend
    Parameters
    ----------
    data : DataFrame
        The dataframe which contains all exogenous variables.
    cols : list
        A list containing the columns which should be
        analysed
    spike_dict : dict
        A dictionary having all columns as the keys and the
        potential spike as the value
    city_name : str
        A string to save the resulting png properly
    Returns
    -------
    None.
    """
    fig, axs = plt.subplots(nrows=4, ncols=5, sharex=True, sharey=True,
                            figsize=(20, 20))
    plt.subplots_adjust(right=None)
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor="none", top=False, bottom=False,
                    left=False, right=False)
    plt.grid(False)
    plt.xlabel("Lags", fontsize=22, labelpad=20)
    plt.ylabel("Correlation", fontsize=22, labelpad=50)
    axs = axs.ravel()
    max_lags = round(np.nanmax(list(spike_dict.values())))
    for i, col in enumerate(cols):
        series = data.loc[:, col].copy()
        sm.graphics.tsa.plot_acf(series.values.squeeze(),
                                 lags=max_lags, ax=axs[i], missing="drop")
        axs[i].set_title(col, fontsize=12)
        axs[i].tick_params(axis="both", labelsize=16)
        if not np.isnan(spike_dict[col]):
            axs[i].axvline(spike_dict[col], -1, 1, color="red",
                           label="Periodicity: {}".format(spike_dict[col]))
            axs[i].legend(loc="upper center", prop={'size': 16})
    fig.tight_layout()
    fig.savefig(r"{}/{}_autocorrelation_function.png".format(output_path,
                                                             city_name),
                bbox_inches="tight")
```

From the ACF plots below, we can extract a lot of useful information. First of all, we can clearly see that for all columns where the Fourier transforms find a significant seasonality, we also find confirming picture. This is because we see a peaking and significant autocorrelation at the lag which was found by the power-plot.

Additionally, we find some variables (e.g. ndvi_nw) which exhibit a constant significant positive autocorrelation. This is a sign of non-stationarity, which will be addressed in the next section which will be dealing of stochastic and deterministic trends.

![](/assets/post_images/dengai/picture1_7.png)

In order to get rid of the seasonal component, we decompose each seasonality-affected feature into its unaffected version its seasonality component and trend component. This is done by the STL decomposition which was developed by Cleveland, McRae & Terpenning (1990). STL is an acronym for "Seasonal and Trend decomposition using Loess", while Loess is a method for estimating non-linear relationships.

The following code snippet decomposes the relevant time series, and subtracts (given that we face additive seasonalities) the seasonality and the trend from the time series.

```python
def season_trend(self, data, cols, spike_dict):
        """
        This method decomposes the time series by removing
        (subtracting) the modelled seasonality and trend.
        Parameters
        ----------
        data : DataFrame
            A dataframe containing the relevant time series
        cols : list
            A list which specifies all potentially affected columns
        spike_dict : dict
            A dictionary stating the significant seasonality for
            each column
        Returns
        -------
        data : Dataframe
            After decomposing and 'cleaning', we put the variables
            back into the dataframe which is returned
        """
        for col in cols:
            period = spike_dict[col]
            time_series = data.loc[:, col]
            if not np.isnan(period):
                res = STL(time_series, period=int(spike_dict[col])+1).fit()
                adjusted_series = time_series - res.seasonal
                data.loc[:, col] = adjusted_series

        return data
```

## Deterministic Trends

One more obvious way to breach the assumptions of covariance stationarity is if the series has a deterministic trend. It is important to stress the difference between a deterministic and not a stochastic trend (unit root). Whereas it is possible to model and remove a deterministic trend, this is not possible with a stochastic trend, given its unpredictable and random behavior.

A deterministic trend is the simplest form of a non-stationary process and time series which exhibit such a trend can be decomposed into three components:

![](/assets/post_images/dengai/picture1_71.png)

The most common type of trend is a linear trend. It is relatively straight forward to test for such a trend and remove it, if one is found. We apply the original Mann-Kendall test, which does not consider seasonal effects, which we already omitted in the part above. If a trend is found, it is simply subtracted from the time series. These steps are completed in the method shown below.

```python
def trend_detecter(self, data, cols):
      """
      This method tests for a deterministic trend using the
      Mann-Kendall test. If the test is found to be significant,
      the trend is removed (subtracted).
      Parameters
      ----------
      data : DataFrame
          A dataframe containing all the relevant columns
      cols : list
          A list of column names for which we apply the test
      Returns
      -------
      no_nan_data : DataFrame
          A dataframe with the potentially removed trend series
      trend_dict : dict
          A dictionary containing the information of the detrending
      """
      trend_dict = {}
      for col in cols:
          trend_dict[col] = {}

          time_series = data.loc[:, col]
          result = mk.original_test(time_series)
          trend_dict[col]["pre_trend"] = result.trend

          if result.trend != "no trend":
              d_trend = [(result.intercept + result.slope * x)
                         for x in np.arange(len(time_series))]
              trend_dict[col]["intercept"] = result.intercept
              trend_dict[col]["slope"] = result.slope

              adj_time_series = time_series - d_trend
              result = mk.original_test(adj_time_series)
              trend_dict[col]["post_trend"] = result.trend
              data.loc[:, col] = adj_time_series

      no_nan_data = data.dropna(subset=cols).reset_index(drop=True)
      return no_nan_data, trend_dict
```

The result can be viewed here. As we can see, most time series exhibited a linear trend, which was then removed.

```python
| ndvi_ne    | ndvi_nw                | ndvi_se                 | ndvi_sw  | precipitation_amt_mm    | reanalysis_air_temp_k | reanalysis_avg_temp_k | reanalysis_dew_point_temp_k | reanalysis_max_air_temp_k | reanalysis_min_air_temp_k | reanalysis_precip_amt_kg_per_m2 | reanalysis_relative_humidity_percent | reanalysis_sat_precip_amt_mm | reanalysis_specific_humidity_g_per_kg | reanalysis_tdtr_k     | station_avg_temp_c     | station_diur_temp_rng_c | station_max_temp_c     | station_min_temp_c      | station_precip_mm     |                     |
|------------|------------------------|-------------------------|----------|-------------------------|-----------------------|-----------------------|-----------------------------|---------------------------|---------------------------|---------------------------------|--------------------------------------|------------------------------|---------------------------------------|-----------------------|------------------------|-------------------------|------------------------|-------------------------|-----------------------|---------------------|
| Trend      | decreasing             | decreasing              | no trend | decreasing              | decreasing            | increasing            | increasing                  | increasing                | increasing                | increasing                      | decreasing                           | decreasing                   | decreasing                            | increasing            | increasing             | increasing              | decreasing             | decreasing              | increasing            | increasing          |
| Slope      | -8.804180148944889e-05 | -0.00010508442895612228 | NA       | -1.9969121004566154e-05 | -0.003975390783667321 | 0.00079050539377047   | 0.0007647709685904174       | 0.0004375871000444891     | 0.000685165752960625      | 0.0006468026767682056           | -0.012109224127889808                | -0.0017669399624736161       | -0.003975390783667321                 | 0.0004258943155599426 | 0.00032172583198841135 | 0.0001547004468416762   | -0.0009195402298850562 | -0.00030887303593728855 | 0.0006413032891119354 | 0.00986137183746052 |
| Intercept  | 0.10424117470021918    | 0.12204044630128306     | NA       | 0.17419584980022826     | 31.7450108145569      | 298.76309596706034    | 298.8943108197956           | 294.9033123255635         | 301.01619501182535        | 297.00822681183297              | 32.6573650987486                     | 79.43547713504724            | 31.7450108145569                      | 16.345457478691138    | 2.292595026925666      | 26.96210500459904       | 7.149425287356321      | 31.81129219530351       | 22.338233161530237    | 19.02484232581643   |
| Post Trend | no trend               | no trend                | NA       | no trend                | no trend              | no trend              | no trend                    | no trend                  | no trend                  | no trend                        | no trend                             | no trend                     | no trend                              | no trend              | no trend               | no trend                | no trend               | no trend                | no trend              | no trend            |
```

Even though we removed a deterministic trend, this did not ensure that our time series are actually stationary now. That is because what works for a deterministic trend does not work for a stochastic trend, meaning that the trend-removing we just did does not ensure stationary of unit-roots.

We therefore have to explicitly test for a unit-root in every time series.

## Stochastic Trends - Unit roots

A unit root process is the generalization of the classic random walk, which is defined as the succession of random steps. Given this definition, the problem of estimating such a time series are obvious. Furthermore, a unit root process violates the covariance stationarity assumptions of not being dependent on time.

To see why that is the case, we assume an autoregressive model where today's value only depends on yesterday's value and an error term.

![](/assets/post_images/dengai/picture1_8.png)

If we parameter a_1 would now be equal to one, the process would simplify to

![](/assets/post_images/dengai/picture1_9.png)

By repeated substitution we could also write this expression as:

![](/assets/post_images/dengai/picture1_10.png)

When now calculating the variance of y_t, we face a variance which is positively and linearly dependent on time, which violates the second covariance stationarity rule.

![](/assets/post_images/dengai/picture1_11.png)

This would have not been the case if a_1 would be smaller than one. That is also basically what is tested in an unit-root test. Arguably the most well-known test for an unit root is the Augmented Dickey Fuller (ADF) test. This test has the null hypothesis of having a unit root present in an autoregressive model. The alternative is normally that the series is stationary or trend-stationary. Given that we already removed a (linear) trend, we assume that the alternative is a stationary series.

In order to be technically correct, it is to be said that the ADF test is not directly testing that a_1 is equal to zero, but rather looks at the characteristic equation. The equation below illustrates what is meant by that:

![](/assets/post_images/dengai/picture1_12.png)

We can see that the difference to the equation before is that we do not look at the level of y_t, but rather at the difference of y_t. Capital Delta represent here the difference operator. The ADF is now testing whether the small delta operator is equal to zero. If that would not be the case, then the difference between yesterday's and tomorrow's value would depend on yesterday's value. That would mean if the today's value is high, the difference between today's and tomorrow's value will also be large which is a self-enforcing and explosive process which clearly depends on time and therefore breaks the assumptions of covariance stationarity.

In case of a significant unit-root (meaning a pvalue above 5%), we difference the time series as often as necessary until we find a stationary series. All of that is done through the following two methods.

```python
def dickey_fuller_test(self, data, cols):
    """
    Method to test certain rows from a dataframe whether
    an unit root is present through the ADF test
    Parameters
    ----------
    data : Dataframe
        A dataframe which contains all series we would like to
        test
    cols : list
        A list containing all columns names for which we would
        like to conduct the test for.
    Returns
    -------
    adf_dict : dict
        Dictionary containing the test result for every series.
    """
    adf_dict = {}
    for col in cols:
        time_series = data.loc[:, col].dropna()
        result = adfuller(time_series, autolag="AIC", regression="c")
        adf_dict[col] = result[1]
    return adf_dict

def diff_sign_columns(self, data, cols, adf_dict):
    """
    This method differences the time series if a non significant
    dickey fuller test is shown. This is done as long as the
    adf is not significant.
    Parameters
    ----------
    data : Dataframe
        A dataframe containing all the time series we would like to test
    cols : list
        List of column names we would like to test
    adf_dict : dict
        dictionary containing the test results of the dickey fuller test
    Returns
    -------
    data : DataFrame
        A dataframe with the now potentially differenced series
    adf_dict : dict
        A dictionary with the now significant dickey fuller tests
    number_of_diff : dict
        A dictionary telling how often each series was differenced.
    """
    number_of_diff = {}
    for col in cols:
        pvalue = adf_dict[col]
        time_series = data.loc[:, col].dropna()
        while pvalue > 0.05:
            time_series = time_series.diff(periods=1)
            pvalue = adfuller(time_series.dropna(),
                              autolag="AIC",
                              regression="c")[1]
            number_of_diff[col] = sum(time_series.isna())
        adf_dict[col] = pvalue
        data.loc[:, col] = time_series

    return data, adf_dict, number_of_diff
```

The following table shows that we do not find any significant ADF test, meaning that no differencing was needed and that no series exhibited a significant unit root.

## Finishing up
Last but not least we take a look at our processed time series. It is nicely visible that none of the time series are trending anymore and they do not exhibit significant seasonality anymore.

![](/assets/post_images/dengai/picture1_13.png)

Additionally we take a look at how the distributions of all of the series look. It is important to note that there are no distributional assumptions of the feature variables when it comes to forecasting. That means that even if we find highly skewed variables, it is not necessary to apply any transformation.

![](/assets/post_images/dengai/picture1_14.png)

After sufficiently transforming all exogenous variables, it is now time to shift our attention on the forecasting procedure of both cities.
