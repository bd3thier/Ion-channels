Kalman filter tests
================
Berenice
April 27, 2020

## Apply Kalman filter - change Q and H

``` r
# Load packages
library(knitr)
library(rmarkdown)
library(tidyverse)
```

    ## -- Attaching packages -------------------------------------------------------------------------------- tidyverse 1.3.0 --

    ## v ggplot2 3.3.2     v purrr   0.3.4
    ## v tibble  3.0.3     v dplyr   1.0.0
    ## v tidyr   1.1.0     v stringr 1.4.0
    ## v readr   1.3.1     v forcats 0.5.0

    ## -- Conflicts ----------------------------------------------------------------------------------- tidyverse_conflicts() --
    ## x dplyr::filter() masks stats::filter()
    ## x dplyr::lag()    masks stats::lag()

``` r
library(xts)
```

    ## Loading required package: zoo

    ## 
    ## Attaching package: 'zoo'

    ## The following objects are masked from 'package:base':
    ## 
    ##     as.Date, as.Date.numeric

    ## 
    ## Attaching package: 'xts'

    ## The following objects are masked from 'package:dplyr':
    ## 
    ##     first, last

``` r
#library(dygraphs)
library(KFAS)
library(ggplot2)
#library(plotly)
library(seewave)
```

    ## 
    ## Attaching package: 'seewave'

    ## The following object is masked from 'package:readr':
    ## 
    ##     spec

``` r
library(signal)
```

    ## 
    ## Attaching package: 'signal'

    ## The following object is masked from 'package:seewave':
    ## 
    ##     unwrap

    ## The following object is masked from 'package:dplyr':
    ## 
    ##     filter

    ## The following objects are masked from 'package:stats':
    ## 
    ##     filter, poly

``` r
library(reticulate)
use_python("C:/Users/beren/anaconda3/envs/TensorFlow-GPU")
use_virtualenv ('C:/Users/beren/Ion-channel/notebooks/R code/.env')

#theme_set(theme_light())
```

``` python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns
from scipy.signal import find_peaks
from sklearn import metrics
```

## Identify peaks

We need to create to functions:

  - set\_param\_find\_peaks: We want to use scipy “find\_peaks” and
    detect all the peaks corresponding to an open channel (even if that
    means having a lot of noise), so the highest Recall. For that, we
    can tune height, threshold, distance, prominence, width, relative
    height and plateau size arguments.

  - find\_peaks\_signal: Once we have found the best parameters, we want
    to export the list of peak as a feature of train and test datasets
    (bool, True when a peak is detected).

<!-- end list -->

``` python
# Function to find peaks and return accuracy
from scipy.signal import find_peaks

def set_param_find_peaks(df, signal, channel_array, height=None, threshold=None, distance=None, 
                         prominence=None, width=None, wlen=None, rel_height=0.5, plateau_size=None):
    # find peaks with given parameters
    x = signal
    peaks, properties = find_peaks(x, height=height, threshold=threshold, distance=distance,  
                                  prominence=prominence, width=width, wlen=wlen, rel_height=rel_height, 
                                  plateau_size=plateau_size)
    plt.plot(x)
    plt.plot(peaks, x[peaks], "x")
    plt.plot(np.zeros_like(x), "--", color="gray")
    plt.show()
    print('Number of peaks found:', len(peaks))
    
    #Assess if all peaks have been found
    df['peak_detected']= pd.Index(range(0,len(df))).isin(peaks)
    df['is_channel_open'] = channel_array.astype(int) != 0
    detected = df.peak_detected.sum()
    open_channels = df.is_channel_open.sum()
    print('actual open channels:', open_channels)
    
    metrics.roc_auc_score(df['is_channel_open'], df['peak_detected'])
    conf_matr = metrics.confusion_matrix(df['is_channel_open'], df['peak_detected'])

    tn, fp, fn, tp = conf_matr.ravel()
    print('False positives:', fp) 
    print('True positives:', tp) 
    print('Precision:', np.round(100*tp /(tp+fp), 2), '%')
    print('False negatives:', fn) 
    print('True negatives:', tn) 
    print('Recall:', np.round(100*tp /(tp+fn), 2), '%')
    
    return df, tn, fp, fn, tp
```

``` python
# Function to find peaks: adds a 'peak_detected' column to the dataframe with bool

def find_peaks_signal(df, signal, height=None, threshold=None, distance=None, prominence=None,
                      width=None, wlen=None, rel_height=0.5, plateau_size=None):
    x = signal
    peaks, properties = find_peaks(x, height=height, threshold=threshold, distance=distance, 
                                   prominence=prominence, width=width, wlen=wlen, rel_height=rel_height,
                                   plateau_size=plateau_size)
    df['peak_detected']= 0
    df['peak_detected']= pd.Index(range(0,len(df))).isin(peaks)
    return df.head()
```

## Apply Kalman filter

``` r
#Read in the training data

train <- read_csv('C:/Users/beren/Ion-channel/data/external/train_clean.csv')
```

    ## Parsed with column specification:
    ## cols(
    ##   time = col_double(),
    ##   signal = col_double(),
    ##   open_channels = col_double()
    ## )

``` r
test <- read_csv('C:/Users/beren/Ion-channel/data/external/test_clean.csv')
```

    ## Parsed with column specification:
    ## cols(
    ##   time = col_double(),
    ##   signal = col_double()
    ## )

``` r
trainKF_download <- read_csv('C:/Users/beren/Ion-channel/data/external/train_kalman.csv')
```

    ## Parsed with column specification:
    ## cols(
    ##   time = col_double(),
    ##   signal = col_double(),
    ##   open_channels = col_double()
    ## )

``` r
testKF_download <- read_csv('C:/Users/beren/Ion-channel/data/external/test_kalman.csv')
```

    ## Parsed with column specification:
    ## cols(
    ##   time = col_double(),
    ##   signal = col_double()
    ## )

We can tweak parameters Q and H. I did not save all the attempts.

``` r
#Function for applying kalman filter
kf_smooth <- function(signal, Q, H) {
  kf <- SSModel(signal ~ SSMtrend(1, Q = Q), H = H)
  out <- KFS(kf)
  alphahat <- out$alphahat
  return(alphahat)
}

#Apply KF
trainKF <- train
testKF <- test
trainKF <- train %>%  mutate(
  alphahat = kf_smooth(signal, Q = 0.1, H = 0.1)
) %>% select(time, signal = alphahat, open_channels)
testKF <- test %>%  mutate(
  alphahat = kf_smooth(signal, Q = 0.001, H = 0.1)
) %>% select(time, signal = alphahat)
```

## Make plots

``` r
fx_plot <- function(df, sample_start = 0, sample_stop = nrow(df), subsample=1){
  data <-
          df %>%
                  mutate(sample = row_number()) %>%
                  dplyr::filter(sample %in% seq(sample_start, sample_stop, subsample))

  plot <-
          data %>%
                  ggplot() +
                  geom_line(aes(sample, signal, col = 'signal')) +
                  geom_line(aes(sample, open_channels)) 

  
  return(plot)
}
```

``` r
start = 100000
stop = 200000
#ggplotly(
  fx_plot(train, start,stop, 1) 
```

![](Kalman-filter_testHQ_files/figure-gfm/unnamed-chunk-8-1.png)<!-- -->

``` r
#  )

#ggplotly(
  fx_plot(trainKF, start,stop, 1)
```

    ## Don't know how to automatically pick scale for object of type ts. Defaulting to continuous.

![](Kalman-filter_testHQ_files/figure-gfm/unnamed-chunk-8-2.png)<!-- -->

``` r
  #)

#ggplotly(
  fx_plot(trainKF_download, start,stop, 1)
```

![](Kalman-filter_testHQ_files/figure-gfm/unnamed-chunk-8-3.png)<!-- -->

``` r
  #)
```

Based on this preliminary analysis, the dataset with Klaman filters
downloaded from
[michaln](https://www.kaggle.com/michaln/data-without-drift-with-kalman-filter)
could have even less noise, but with the current settings (Q = 0.001, H
= 0.1) we are missing a few peaks. We will test different Hs and Qs and
use scipy.find\_peak on the new signal. We will then check the accuracy
of the peaks detected (do they match an open channel).

## Remove line noise with notch filter

We want to plot the FFT to visualize the Fourier transform before/after
notch filter.

``` r
plot_fft <- function(signal, fs){
  y <- fft(signal)
  y.tmp <- Mod(y)   
  y.ampspec <- y.tmp[1:(length(y)/2+1)]
  y.ampspec[2:(length(y)/2)] <- y.ampspec[2:(length(y)/2)] * 2
  f <- seq(from=0, to=fs/2, length=length(y)/2+1)
  p <- plot(f, y.ampspec, type="h", xlab="Frequency (Hz)", ylab="Amplitude Spectrum", xlim=c(0, 500), ylim=c(0, 150000))
  
  return(p)
}

notch_filt <- function(signal, from = 49.9, to=50.1, fs){
  # notch filter to remove 50 Hz electrical noise from ion channel signal
  p1 <- plot_fft(signal, fs)
  p1
  signal_filt <- bwfilter(signal, 10000, from= from, to = to, bandpass = NULL)
  p2 <- plot_fft(signal_filt, fs)
  p2
  return(signal_filt)
}
```

``` r
trainKF$signal <- notch_filt(trainKF$signal, 49.9, 50.1, 10000)
```

![](Kalman-filter_testHQ_files/figure-gfm/unnamed-chunk-10-1.png)<!-- -->![](Kalman-filter_testHQ_files/figure-gfm/unnamed-chunk-10-2.png)<!-- -->

``` r
write.csv(trainKF,"../data/interim/trainKF.csv", row.names = FALSE)
```

``` python
import pandas as pd
pd_trainKF = pd.read_csv("../data/interim/trainKF.csv")
pd_trainKF.head()
```

    ##      time    signal  open_channels
    ## 0  0.0001 -2.755737              0
    ## 1  0.0002 -2.763275              0
    ## 2  0.0003 -2.691560              0
    ## 3  0.0004 -2.917845              0
    ## 4  0.0005 -2.936068              0

``` python
pd_trainKF, tn, fp, fn, tp = set_param_find_peaks(pd_trainKF, pd_trainKF.signal, pd_trainKF.open_channels,
                                                  height=-3, threshold=None, distance=None, prominence=0.05,
                                                  width=None, wlen=None, rel_height=0.5, plateau_size=None)
```

    ## Number of peaks found: 887789
    ## actual open channels: 3759848
    ## False positives: 192043
    ## True positives: 695746
    ## Precision: 78.37 %
    ## False negatives: 3064102
    ## True negatives: 1048109
    ## Recall: 18.5 %

![](Kalman-filter_testHQ_files/figure-gfm/unnamed-chunk-13-1.png)<!-- -->

``` r
trainKF <- train %>%  mutate(
  alphahat = kf_smooth(signal, Q = 0.1, H = 1)
) %>% select(time, signal = alphahat, open_channels)
```

``` r
start = 0
stop = 200000
#ggplotly(
  fx_plot(trainKF, start,stop, 1)#)
```

    ## Don't know how to automatically pick scale for object of type ts. Defaulting to continuous.

![](Kalman-filter_testHQ_files/figure-gfm/unnamed-chunk-15-1.png)<!-- -->

``` r
trainKF$signal <- notch_filt(trainKF$signal, 49.9, 50.1, 10000)
```

![](Kalman-filter_testHQ_files/figure-gfm/unnamed-chunk-16-1.png)<!-- -->![](Kalman-filter_testHQ_files/figure-gfm/unnamed-chunk-16-2.png)<!-- -->

``` r
write.csv(trainKF,"../data/interim/trainKF.csv", row.names = FALSE)
```

``` python
import pandas as pd
pd_trainKF = pd.read_csv("../data/interim/trainKF.csv")
pd_trainKF.head()
```

    ##      time    signal  open_channels
    ## 0  0.0001 -2.773977              0
    ## 1  0.0002 -2.775938              0
    ## 2  0.0003 -2.771239              0
    ## 3  0.0004 -2.804309              0
    ## 4  0.0005 -2.805220              0

``` python
pd_trainKF, tn, fp, fn, tp = set_param_find_peaks(pd_trainKF, pd_trainKF.signal, pd_trainKF.open_channels,
                                                  height=-2.2, threshold=-2.2, distance=None, prominence=0.01,
                                                  width=None, wlen=None, rel_height=0.5, plateau_size=None)
```

    ## Number of peaks found: 368959
    ## actual open channels: 3759848
    ## False positives: 51
    ## True positives: 368908
    ## Precision: 99.99 %
    ## False negatives: 3390940
    ## True negatives: 1240101
    ## Recall: 9.81 %

![](Kalman-filter_testHQ_files/figure-gfm/unnamed-chunk-19-1.png)<!-- -->

``` r
testKF <- test %>%  mutate(
  alphahat = kf_smooth(signal, Q = 0.1, H = 1)
) %>% select(time, signal = alphahat)
```

``` r
testKF$signal <- notch_filt(testKF$signal, 49.9, 50.1, 10000)
```

![](Kalman-filter_testHQ_files/figure-gfm/unnamed-chunk-21-1.png)<!-- -->![](Kalman-filter_testHQ_files/figure-gfm/unnamed-chunk-21-2.png)<!-- -->

``` r
write.csv(testKF,"../data/interim/testKF.csv", row.names = FALSE)
```
