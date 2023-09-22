# Project Overview
42 people with early Parkinson's disease (PPD) were used in this experiment to collect voice measures and demographic data for the "po2_data.csv" dataset. The dataset is separate from the first dataset and is only available to PPD participants. A different set of attributes that were taken out of speech samples correlate to each record. Age, gender, test time, and voice-related measures including jitter, shimmer, RPDE, DFA, and PPE are all included in the dataset. A thorough assessment of motor symptoms and total impairment status in people with early-stage Parkinson's disease is provided by the inclusion of Motor UPDRS scores (ranging from 0 to 108) and Total UPDRS scores (ranging from 0 to 176).

The two main goals of this project are to predict the Total UPDRS score for a thorough assessment of overall disability in people with early-stage Parkinson's disease and the Motor UPDRS score for a precise assessment of motor symptoms. For better patient outcomes, these goals seek to improve clinical decision-making, treatment planning, and disease management.

## Key Steps and Insights

The project unfolds through a series of steps. Check the Python code name named Parkinson_Disease_Feature_Selection.py
### 1. Data Loading and Preprocessing

The dataset has a shape of (5875, 22), indicating 5875 instances and 22 features.
![Image Alt Text](Columns_description.png)


#### 1.2 Data Information

The information about the dataset was obtained using the `.info()` method. It confirmed that the dataset contains no missing values, and all columns have either `float64` or `int64` data types.

#### 1.3 Duplicate Rows

Duplicate rows were checked using the `.duplicated()` method. No duplicate rows were found in the dataset.

## 2. Exploratory Data Analysis
Next, we will explore the data. We first checked the proportion of males and females. As shown in the figure below, almost two-thirds of people were male who suffered from Parkinson's Disease.
![Image Alt Text](Bar_diagram.png)

### 2.1 Measure of Central Tendency
The dataset's central tendency measurements, mean values, median values, and standard deviations provide essential insights into its features. The average age and Motor UPDRS score indicate an even distribution, while larger values indicate more dispersion. Understanding these patterns is crucial for interpreting and modelling.

| Feature          | Mean        | Median (50%) | Standard Deviation |
|------------------|-------------|--------------|--------------------|
| subject#         | 21.494128   | 22.000000    | 12.372279          |
| age              | 64.804936   | 65.000000    | 8.821524           |
| sex              | 0.317787    | 0.000000     | 0.465656           |
| test_time        | 92.863722   | 91.523000    | 53.445602          |
| motor_updrs      | 21.296229   | 20.871000    | 8.129282           |
| total_updrs      | 29.018942   | 27.576000    | 10.700283          |
| jitter(%)        | 0.006154    | 0.004900     | 0.005624           |
| jitter(abs)      | 0.000044    | 0.000035     | 0.000036           |
| jitter(rap)      | 0.002987    | 0.002250     | 0.003124           |
| jitter(ppq5)     | 0.003277    | 0.002490     | 0.003732           |
| jitter(ddp)      | 0.008962    | 0.006750     | 0.009371           |
| shimmer(%)       | 0.034035    | 0.027510     | 0.025835           |
| shimmer(abs)     | 0.310960    | 0.253000     | 0.230254           |
| shimmer(apq3)    | 0.017156    | 0.013700     | 0.013237           |
| shimmer(apq5)    | 0.020144    | 0.015940     | 0.016664           |
| shimmer(apq11)   | 0.027481    | 0.022710     | 0.019986           |
| shimmer(dda)     | 0.051467    | 0.041110     | 0.039711           |
| nhr              | 0.032120    | 0.018448     | 0.059692           |
| hnr              | 21.679495   | 21.920000    | 4.291096           |
| rpde             | 0.541473    | 0.542250     | 0.100986           |
| dfa              | 0.653240    | 0.643600     | 0.070902           |
| ppe              | 0.219589    | 0.205500     | 0.091498           |

### 2.2 Histogram with KDE plots

The histogram plots suggest that most data follow normal distribution with most of them being right skewed like shimmering and jitter columns. 

![Image Alt Text](Bar_diagram.png)

### 2.3 Confidence Interval
The confidence intervals for various dataset characteristics are shown in the table below. These confidence intervals provide us a range where the genuine population parameter (mean) is most likely to fall. The feature name, its mean value, the lower and upper confidence interval bounds, as well as the table's feature name are all included.
----------------+------------------------+------------------------+------------------------+
|    Feature     |          Mean          |        CI_Lower        |        CI_Upper        |
+----------------+------------------------+------------------------+------------------------+
|   test_time    |   92.86372204595745    |   91.49679408661369    |    94.2306500053012    |
|      age       |   64.80493617021277    |   64.57931634424544    |    65.0305559961801    |
|  total_updrs   |   29.01894228085106    |   28.745271187609134   |   29.292613374092987   |
|      hnr       |   21.679495148936173   |   21.569745810660194   |   21.78924448721215    |
|    subject#    |   21.494127659574467   |   21.177693491907018   |   21.810561827241916   |
|  motor_updrs   |   21.29622854468085    |   21.088313547525818   |   21.504143541835884   |
|      dfa       |   0.6532396561702127   |   0.6514262624295218   |   0.6550530499109036   |
|      rpde      |   0.5414729327659574   |   0.5388901141435086   |   0.5440557513884062   |
|      sex       |   0.3177872340425532   |   0.3058775905413794   |   0.329696877543727    |
|  shimmer(abs)  |  0.31095999999999996   |  0.30507101835371936   |  0.31684898164628056   |
|      ppe       |  0.21958917906382977   |  0.21724900803481476   |  0.22192935009284478   |
|  shimmer(dda)  |  0.05146694127659574   |  0.050451280290145575  |  0.05248260226304591   |
|   shimmer(%)   |  0.03403521702127659   |  0.03337445545752784   |  0.03469597858502534   |
|      nhr       |  0.032119817531914896  |  0.030593123504602728  |  0.033646511559227064  |
| shimmer(apq11) |  0.027480861276595745  |  0.026969694409634617  |  0.027992028143556873  |
| shimmer(apq5)  |  0.02014418382978723   |  0.019717986072310317  |  0.020570381587264146  |
| shimmer(apq3)  |  0.017155703829787233  |  0.016817150505973894  |  0.017494257153600572  |
|  jitter(ddp)   |  0.00896167659574468   |  0.008721990788653287  |  0.009201362402836071  |
|   jitter(%)    |  0.006153761702127659  | 0.0060099160987435295  |  0.006297607305511788  |
|  jitter(ppq5)  | 0.0032768510638297873  |  0.003181413062163381  | 0.0033722890654961936  |
|  jitter(rap)   |  0.002987184680851064  |  0.002907289697222909  | 0.0030670796644792185  |
|  jitter(abs)   | 4.4027118297872336e-05 | 4.3106805735977006e-05 | 4.4947430859767666e-05 |
+----------------+------------------------+------------------------+------------------------+



