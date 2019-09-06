# hist.fct
# gnuplot macro for providing a functionality similar to the hist() function in octave
# Note, that the variables binwidth and binstart has to be set before calling this function
# AUTHOR: Hagen Wierstorf

# set width of single bins in histogram
set boxwidth 0.9*binwidth
# set fill style of bins
set style fill solid 0.5
# define macro for plotting the histogram
hist = 'u (binwidth*(floor(($1-binstart)/binwidth)+0.5)+binstart):(1.0) smooth freq w boxes'
