from curses import REPORT_MOUSE_POSITION
from os import renames
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, pacf, plot_pacf
import pmdarima as pm
import datetime
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report

st.title('Demand Forecasting')
st.header('Upload data to analyse and forecast')
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()

    # Can be used wherever a "file-like" object is accepted:
    data = pd.read_csv(uploaded_file)

    index_col = st.selectbox("Select date column", data.columns.to_list())
    data[index_col] = pd.to_datetime(data[index_col])
    data = data.set_index(data[index_col])
    st.write(data)

customer = st.selectbox("Select Customer",  data.columns.to_list())
    #data = data.resample('1h').mean().replace(0., np.nan)
earliest_time = data.index.min()
df=data[[customer]]

df_list = []

for label in df:

    ts = df[label]

    start_date = min(ts.fillna(method='ffill').dropna().index)
    end_date = max(ts.fillna(method='bfill').dropna().index)

    active_range = (ts.index >= start_date) & (ts.index <= end_date)
    ts = ts[active_range].fillna(0.)

    tmp = pd.DataFrame({'time_series': ts})
    date = tmp.index
    
    tmp['date'] = date
    #tmp['consumer_id'] = label
    tmp['hour'] = date.hour
    tmp['day'] = date.day
    tmp['day_of_week'] = date.dayofweek
    tmp['month'] = date.month

        #stack all time series vertically
    df_list.append(tmp)

time_df = pd.concat(df_list).reset_index(drop=True)


data = time_df[['time_series','date']]
data = data.set_index(['date'])

y = time_df[['time_series']]
data.index = pd.DatetimeIndex(data.index.values,
                                freq=data.index.inferred_freq)
y = y.set_index(data.index)


my_button = st.sidebar.radio("Steps", ('Analysis', 'Model training and forecast')) 


if my_button == 'Analysis': 

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Profile Report", "EDA", "Decomposition", "ADF Test", "ACF", "PACF"])


    with tab1:
        pr = data.profile_report()
        st_profile_report(pr)

    with tab2:
        plot_type = st.radio('Plot Type', ['Single Plot', 'Grouped Plot'])
        col1, col2, col3 = st.columns(3)

        
        if plot_type == 'Single Plot':
            with col1:
                xaxis = st.selectbox("x_axis", time_df.columns.to_list())
            with col2:
                yaxis = st.selectbox('y-axis', time_df.columns.to_list())
        else:
            with col1:
                xaxis = st.selectbox("x_axis", time_df.columns.to_list())
            with col2:
                yaxis = st.selectbox('y-axis', time_df.columns.to_list())
            with col3:
                groupby = st.selectbox("Group_by", time_df.columns.to_list())

        plot_button = st.button("Plot")
        if plot_button:
            if plot_type == 'Single Plot':
                fig, ax = plt.subplots()
                sns.lineplot(data=time_df, 
                            x=xaxis, 
                            y=yaxis, 
                            #hue='day_of_week', 
                            legend='full',
                            palette='husl').set_title('')

                ax.set_xlabel(xaxis, fontdict={"color":  "black", "size": 13} )
                ax.set_ylabel(yaxis, fontdict={"color":  "black", "size": 13})
                ax.tick_params(axis = 'both', which = 'major', labelsize = 10)
                st.pyplot(fig)
            else:
                fig, ax = plt.subplots()
                sns.lineplot(data=time_df, 
                            x=xaxis, 
                            y=yaxis, 
                            hue=groupby, 
                            legend='full',
                            palette='husl').set_title('Power usage per consumers')

                ax.set_xlabel(xaxis, fontdict={"color":  "black", "size": 13} )
                ax.set_ylabel(yaxis, fontdict={"color":  "black", "size": 13})
                ax.tick_params(axis = 'both', which = 'major', labelsize = 10)
                st.pyplot(fig)

    with tab3:
        st.write("Decomposition")
        decomposition = sm.tsa.seasonal_decompose(y, model = 'additive', period = 12)
        fig = decomposition.plot()
        st.pyplot(fig)

    with tab4:
        # apply ADF and print results
        result = adfuller(y)
        st.subheader('Augmented Dickey-Fuller Test')
        st.write('Statistical Test: {:.4f}'.format(result[0]))
        st.write('P Value: {:.10f}'.format(result[1]))
        st.write('Critical Values:')
        for key, value in result[4].items():
            st.write('\t{}: {:.4f}'.format(key, value))

    with tab5:
        st.subheader('Autocorrelation Plot')
        fig = plot_acf(data, lags=30)
        st.pyplot(fig)

    with tab6:
        st.subheader('Partial Autocorrelation Plot')
        fig = plot_pacf(data, lags=30, method = 'ywm')
        st.pyplot(fig)
elif my_button == 'Model training and forecast':
    st.header('ARIMA/SARIMA')
    Model = st.radio("Select Appropriate Model and Parameters to Train", ('ARIMA', 'SARIMA'))

    col1, col2, col3 = st.columns(3)

    if Model == 'ARIMA':
        with col1:
            p = st.number_input('p', min_value = 0, max_value = 5, value = 1)
            d = st.number_input('d', min_value = 0, max_value = 5, value = 0)
            q = st.number_input('q', min_value = 0, max_value = 5, value = 1)
        with col2:
            step = st.number_input("Number of steps to forecast", min_value=1, max_value=1000, value = 60)
    else:
        with col1:
            p = st.number_input('p', min_value = 0, max_value = 5, value = 1)
            d = st.number_input('d', min_value = 0, max_value = 5, value = 0)
            q = st.number_input('q', min_value = 0, max_value = 5, value = 1)
        with col2:
            P = st.number_input('P', min_value = 0, max_value = 5, value = 1)
            D = st.number_input('D', min_value = 0, max_value = 5, value = 0)
            Q = st.number_input('Q', min_value = 0, max_value = 5, value = 1)
        with col3:
            seasonality = st.selectbox('Periodicity', [6, 12, 24])
            step = st.number_input("Number of steps to forecast", min_value=1, max_value=1000, value = 60)

    Model_train = st.button("Train")

    if Model_train:
        y.index = pd.DatetimeIndex(y.index.values,
                                    freq=y.index.inferred_freq)
        if Model == 'ARIMA':
            model = sm.tsa.statespace.SARIMAX(y, order=(p, d, q), enforce_invertibility=False,
                                            enforce_stationarity=False)
        else:
            model = sm.tsa.statespace.SARIMAX(y, order=(p, d, q), seasonal_order=(P, D, Q, 
                                                seasonality), enforce_invertibility=False,
                                                enforce_stationarity=False)        

        results = model.fit()


        st.subheader('Model Diagnostics')
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Metrics", "Diagnostics", "Fitted Values",
                                                    'Obs vs fit recent', 'Forecast'])

        pred = results.get_prediction(dynamic=False)
        pred_ci = pred.conf_int()

        y_predicted = pred.predicted_mean
        y_true = data['time_series']
        predictions_SARIMA = pd.Series(results.fittedvalues, copy=True)

        with tab1:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("AIC: " ,round(results.aic,1))
            with col2:
                mse = ((y_true[1:] - y_predicted[1:])**2).mean()
                st.metric('MSE', round(mse, 2))
            with col3:
                st.metric('RMSE', round(np.sqrt(mse),2))
            with col4:
                def mean_absolute_percentage_error(y_true, y_pred): 
                    y_true, y_pred = np.array(y_true), np.array(y_pred)
                    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

                st.metric("MAPE: ",round(mean_absolute_percentage_error(y_true[1:]+1, y_predicted[1:]+1),2))
        
        with tab2:
            fig = results.plot_diagnostics(figsize=(15, 8))
            st.pyplot(fig)

        with tab3:
            fitted_data = pd.DataFrame(y)
            fitted_data["fitted"] = predictions_SARIMA
            st.write(fitted_data[1:])

        with tab4:
            st.subheader('Observed vs fitted values (last one week)')

            predictions_SARIMA = pd.DataFrame(results.fittedvalues, copy=True)
            predict = predictions_SARIMA.loc['2014-09-01':]

            series = data.loc['2014-09-01':]
            series = series['time_series']
            ind = series.index

            fig, ax = plt.subplots()
            ax.plot(series,label='observed')
            ax.plot(predict, label = "fitted")
            ax.legend()
            st.pyplot(fig)
        with tab5:
            # Forecasting the future (out of sample) values.
            pred_uc = results.get_forecast(steps=step)
            pred_ci = pred_uc.conf_int()

            d1 = {'Time':pd.date_range(start =data.index.max(),periods = step, freq ='H')}
            df = pd.DataFrame(d1)

            predicted = pd.DataFrame(pred_uc.predicted_mean)
            predicted = predicted.set_index(df['Time'])

            fig, ax = plt.subplots()
            ax.plot(series,label='observed')
            ax.plot(predict, label = "fitted")
            ax.plot(predicted, label = "forecasted", color = 'green')
            plt.legend()
            st.plotly_chart(fig)



        

